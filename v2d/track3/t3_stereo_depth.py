"""FoundationStereo (NVIDIA TAO small 576x960 ONNX) metric depth for preprocessed Track 3 episodes -- host venv, no Docker.

Uses the repo module's own pre/post-processing and TensorRT runner (v2d.foundation_stereo.lib.trt_inference) but
  * builds the TensorRT engine with the TensorRT Python API (pip wheels ship no trtexec), saved under the same
    versioned filename that v2d.foundation_stereo.lib.export_engine.ensure_engine looks for;
  * reads our grayscale rectified JPEGs (the module's image_list_to_depth crashes on 2-D gray frames);
  * optional --backend ort: onnxruntime-gpu (CUDA EP) instead of TensorRT (fallback if the engine build fails).
Output per episode: OUT/<split>/episode_XXXXXX/depth_rect/000000.png  -- uint16 inverse depth, v2d DepthImage
  format (pixel = 65535/(depth_m+1); decode depth_m = 65535/pixel - 1), on the rectified-LEFT grid (K_rect in meta.json)
and depth_rect.json (fx, baseline, timing). GPU REQUIRED (except --dry_run, which checks I/O + preprocessing on CPU).

  source envs/t3-geom.env
  python -I t3_stereo_depth.py --frames /mnt/secondary/v2d/t3/frames --out /mnt/secondary/v2d/t3/depth \
      --model_dir /mnt/secondary/v2d/weights/foundationstereo [--split public evaluation] [--episodes 12 ...]
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

ONNX = "deployable_foundationstereo_small_576x960_v2.0.onnx"


def build_engine_python_api(onnx_path, engine_path, fp16=False, workspace_gb=6):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("TensorRT engine build failed")
    tmp = engine_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(ser)
    os.replace(tmp, engine_path)
    print(f"built {engine_path} in {time.time() - t0:.0f}s (fp16={fp16})", flush=True)


class OrtRunner:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CUDAExecutionProvider"])

    def infer(self, left_bgr, right_bgr):
        from v2d.foundation_stereo.lib.trt_inference import FoundationStereoInference as F
        ln, md = F.preprocess_image(None, left_bgr)
        rn, _ = F.preprocess_image(None, right_bgr)
        d = self.sess.run(None, {"left_image": ln, "right_image": rn})[0][0, 0]
        return F._transform_to_original(None, d, md), md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/mnt/secondary/v2d/t3/frames")
    ap.add_argument("--out", default="/mnt/secondary/v2d/t3/depth")
    ap.add_argument("--model_dir", default="/mnt/secondary/v2d/weights/foundationstereo")
    ap.add_argument("--split", nargs="*", default=["public", "evaluation"])
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--backend", choices=["trt", "ort"], default="trt")
    ap.add_argument("--fp16", action="store_true", help="FP16 engine (module default is FP32)")
    ap.add_argument("--every", type=int, default=1, help="process every k-th frame")
    ap.add_argument("--workspace_gb", type=float, default=18,
                    help="TRT builder workspace; the cost-volume Myelin node needs ~14.4 GB at 576x960 FP32")
    ap.add_argument("--dry_run", action="store_true", help="CPU: check inputs/preprocessing only")
    a = ap.parse_args()

    from v2d.common.datatypes import DepthImage
    from v2d.foundation_stereo.lib.trt_inference import FoundationStereoInference, disparity_to_depth

    onnx_path = f"{a.model_dir}/{ONNX}"
    runner = None
    if not a.dry_run:
        if a.backend == "trt":
            from v2d.foundation_stereo.lib.export_engine import get_engine_path
            engine = get_engine_path(a.model_dir)
            if a.fp16:
                engine = engine.replace(".engine", "_fp16.engine")
            if not os.path.exists(engine):
                build_engine_python_api(onnx_path, engine, fp16=a.fp16, workspace_gb=a.workspace_gb)
            runner = FoundationStereoInference(engine)
        else:
            runner = OrtRunner(onnx_path)

    eps = []
    for split in a.split:
        for d in sorted(glob.glob(f"{a.frames}/{split}/episode_*")):
            if a.episodes and int(d[-6:]) not in a.episodes:
                continue
            eps.append(d)
    for ep_dir in eps:
        meta = json.load(open(f"{ep_dir}/meta.json"))
        fx, base = meta["stereo"]["fx"], meta["stereo"]["baseline_m"]
        od = f"{a.out}/{meta['split']}/episode_{meta['episode']:06d}"
        os.makedirs(f"{od}/depth_rect", exist_ok=True)
        t0, n, stats = time.time(), 0, []
        todo = [i for i in range(0, meta["n_frames"], a.every) if not os.path.exists(f"{od}/depth_rect/{i:06d}.png")]

        def read(i):
            L = cv2.cvtColor(cv2.imread(f"{ep_dir}/left/{i:06d}.jpg", cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            R = cv2.cvtColor(cv2.imread(f"{ep_dir}/right/{i:06d}.jpg", cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            return L, R

        def write(i, depth):
            cv2.imwrite(f"{od}/depth_rect/{i:06d}.png", np.array(DepthImage(depth=depth).to_pil_image(), dtype=np.uint16))
            v = depth[(depth > 0.1) & (depth < 3)]
            return float(np.median(v)) if len(v) else float("nan")

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(3)  # prefetch the next pair + write PNGs while the GPU runs
        nxt = pool.submit(read, todo[0]) if todo else None
        pend, t_inf = [], 0.0
        for k, i in enumerate(todo):
            L, R = nxt.result()
            if k + 1 < len(todo):
                nxt = pool.submit(read, todo[k + 1])
            if a.dry_run:
                ln, md = FoundationStereoInference.preprocess_image(None, L)
                assert ln.shape == (1, 3, 576, 960), ln.shape
                print(f"dry_run {od}: input {L.shape} -> {ln.shape} scale {md['scale']:.3f} pad_w {md['pad_w']}")
                break
            ti = time.time()
            disp, _ = runner.infer(L, R)
            t_inf += time.time() - ti
            depth = disparity_to_depth(disp, fx, base)
            pend.append(pool.submit(write, i, depth))
            n += 1
            if n % 50 == 0:
                print(f"  {od}: {n}/{len(todo)} frames, {(time.time() - t0) / n:.3f} s/frame "
                      f"(infer {t_inf / n:.3f})", flush=True)
        stats = [p.result() for p in pend]
        pool.shutdown()
        if not a.dry_run:
            json.dump(dict(fx=fx, baseline_m=base, K_rect=meta["images"]["left"]["K"], frames_done=n,
                           seconds=round(time.time() - t0, 1), backend=a.backend, fp16=a.fp16,
                           encoding="uint16 inverse depth: depth_m = 65535/pixel - 1 (v2d DepthImage)",
                           median_depth_m=stats), open(f"{od}/depth_rect.json", "w"))
            print(f"{od}: {n} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
