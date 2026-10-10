"""CPU sanity check of the FoundationStereo TAO ONNX on one preprocessed frame (no GPU).

Uses the module's own preprocessing/postprocessing (v2d.foundation_stereo.lib.trt_inference), runs the ONNX with
onnxruntime CPUExecutionProvider, converts to metric depth with the episode's rectified fx/baseline, and compares
with OpenCV SGBM on the same pair. A watchdog aborts if RSS exceeds --max_rss_gb.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -I fs_onnx_cpu_check.py ONNX EP_DIR FRAME OUT_PNG
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("ep_dir")
ap.add_argument("frame", type=int)
ap.add_argument("out_png")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max_rss_gb", type=float, default=6.5)
ap.add_argument("--opt", default="basic", choices=["disable", "basic", "extended", "all"])
a = ap.parse_args()


def watchdog():
    peak = 0
    while True:
        rss = int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 1e6
        peak = max(peak, rss)
        if rss > a.max_rss_gb:
            print(f"ABORT: RSS {rss:.2f} GB > {a.max_rss_gb}", flush=True)
            os._exit(3)
        watchdog.peak = peak
        time.sleep(0.2)


watchdog.peak = 0
threading.Thread(target=watchdog, daemon=True).start()

import onnxruntime as ort  # noqa: E402
from v2d.foundation_stereo.lib.trt_inference import FoundationStereoInference, disparity_to_depth  # noqa: E402

meta = json.load(open(f"{a.ep_dir}/meta.json"))
fx, base = meta["stereo"]["fx"], meta["stereo"]["baseline_m"]
L = cv2.imread(f"{a.ep_dir}/left/{a.frame:06d}.jpg", cv2.IMREAD_GRAYSCALE)
R = cv2.imread(f"{a.ep_dir}/right/{a.frame:06d}.jpg", cv2.IMREAD_GRAYSCALE)
Lb, Rb = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR), cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
pre = FoundationStereoInference.preprocess_image  # does not touch self
ln, md = pre(None, Lb)
rn, _ = pre(None, Rb)
so = ort.SessionOptions()
so.intra_op_num_threads = a.threads
so.inter_op_num_threads = 1
so.enable_cpu_mem_arena = False
so.enable_mem_pattern = False
so.graph_optimization_level = {"disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                               "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                               "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                               "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[a.opt]
t0 = time.time()
sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
t1 = time.time()
print(f"session loaded in {t1 - t0:.1f}s, peak RSS so far {watchdog.peak:.2f} GB", flush=True)
raw = sess.run(None, {"left_image": ln, "right_image": rn})[0]
t2 = time.time()
disp = FoundationStereoInference._transform_to_original(None, raw[0, 0], md)
depth = disparity_to_depth(disp, fx, base)

sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=160, blockSize=5, P1=8 * 25, P2=32 * 25,
                           uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
dsg = sg.compute(L, R).astype(np.float32) / 16
v = (dsg > 2) & (disp > 0.5)
ad = np.abs(disp[v] - dsg[v])
print(f"providers={sess.get_providers()} load {t1 - t0:.1f}s infer {t2 - t1:.1f}s peak RSS {watchdog.peak:.2f} GB")
print(f"disparity range [{disp.min():.1f}, {disp.max():.1f}] px; depth median {np.median(depth[depth > 0]):.3f} m "
      f"(5-95%: {np.percentile(depth[depth > 0], 5):.2f}-{np.percentile(depth[depth > 0], 95):.2f} m)")
print(f"vs SGBM on {v.mean() * 100:.0f}% px: median |d_fs - d_sgbm| = {np.median(ad):.2f} px, "
      f"frac<1px {np.mean(ad < 1):.2f}, frac<3px {np.mean(ad < 3):.2f}")
vis = cv2.applyColorMap(np.clip(255 * (1 - (depth - 0.3) / 1.7), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imwrite(a.out_png, np.vstack([np.hstack([Lb, R[..., None].repeat(3, 2)]),
                                  np.hstack([vis, cv2.applyColorMap(np.clip(dsg * 2, 0, 255).astype(np.uint8),
                                                                    cv2.COLORMAP_TURBO)])])[::2, ::2])
np.save(a.out_png.replace(".png", "_depth.npy"), depth.astype(np.float16))
