"""CPU smoke test of the mask stack on one preprocessed cam_a frame (no GPU):
Grounding DINO (HF transformers, IDEA-Research/grounding-dino-base) text->boxes, then SAM2.1 (official sam2 package,
hiera-large) box->mask.  Optional --sam3: SAM3 (HF transformers Sam3Model) text->masks.
A watchdog aborts if RSS exceeds --max_rss_gb.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -I seg_cpu_check.py EP_DIR FRAME "white pot. pot lid." OUT_PNG
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ep_dir")
ap.add_argument("frame", type=int)
ap.add_argument("prompt")
ap.add_argument("out_png")
ap.add_argument("--weights", default="/mnt/secondary/v2d/weights")
ap.add_argument("--max_rss_gb", type=float, default=6.0)
ap.add_argument("--sam3", action="store_true")
ap.add_argument("--scale", type=float, default=0.5, help="downscale cam_a before inference (CPU speed)")
a = ap.parse_args()
PEAK = [0.0]


def watchdog():
    while True:
        rss = int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 1e6
        PEAK[0] = max(PEAK[0], rss)
        if rss > a.max_rss_gb:
            print(f"ABORT: RSS {rss:.2f} GB > {a.max_rss_gb}", flush=True)
            os._exit(3)
        time.sleep(0.1)


threading.Thread(target=watchdog, daemon=True).start()
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from PIL import Image  # noqa: E402

img_bgr = cv2.imread(f"{a.ep_dir}/cam_a/{a.frame:06d}.jpg")
img_bgr = cv2.resize(img_bgr, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
vis = img_bgr.copy()
print("objects in episode:", json.load(open(f"{a.ep_dir}/meta.json"))["objects"])

with torch.inference_mode():
    if not a.sam3:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        d = f"{a.weights}/grounding_dino/grounding-dino-base"
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(d)
        gd = AutoModelForZeroShotObjectDetection.from_pretrained(d).eval()
        inp = proc(images=img, text=a.prompt.lower(), return_tensors="pt")
        out = gd(**inp)
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.3, text_threshold=0.25,
                                                          target_sizes=[img.size[::-1]])[0]
        t1 = time.time()
        labels = res.get("text_labels", res.get("labels"))
        boxes = res["boxes"].numpy()
        print(f"GroundingDINO-base CPU {t1 - t0:.1f}s: " + "; ".join(
            f"{l} {s:.2f} {np.round(b).astype(int).tolist()}" for l, s, b in zip(labels, res["scores"].tolist(), boxes)))
        del gd
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        t0 = time.time()
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", f"{a.weights}/sam2/sam2.1-hiera-large/sam2.1_hiera_large.pt",
                         device="cpu")
        pred = SAM2ImagePredictor(sam)
        pred.set_image(np.asarray(img))
        for i, b in enumerate(boxes):
            m, s, _ = pred.predict(box=b[None], multimask_output=False)
            m = m[0] > 0
            col = np.array([(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 128, 0)][i % 4], np.uint8)
            vis[m] = (0.5 * vis[m] + 0.5 * col).astype(np.uint8)
            cv2.rectangle(vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col.tolist(), 2)
            cv2.putText(vis, str(labels[i]), (int(b[0]), int(b[1]) - 4), 0, 0.7, col.tolist(), 2)
            print(f"  SAM2.1-L mask {i} '{labels[i]}': {m.sum()} px, score {float(s[0]):.3f}")
        print(f"SAM2.1-L CPU {time.time() - t0:.1f}s")
    else:
        from transformers import Sam3Model, Sam3Processor
        d = f"{a.weights}/sam3/sam3"
        t0 = time.time()
        proc = Sam3Processor.from_pretrained(d)
        model = Sam3Model.from_pretrained(d).eval()
        for i, phrase in enumerate([p.strip() for p in a.prompt.split(".") if p.strip()]):
            inp = proc(images=img, text=phrase, return_tensors="pt")
            out = model(**inp)
            r = proc.post_process_instance_segmentation(out, threshold=0.5, mask_threshold=0.5,
                                                        target_sizes=inp.get("original_sizes").tolist())[0]
            print(f"  SAM3 '{phrase}': {len(r['masks'])} instances, scores {np.round(r['scores'].numpy(), 2).tolist()}")
            col = np.array([(0, 255, 0), (255, 0, 255), (0, 200, 255)][i % 3], np.uint8)
            for m in r["masks"]:
                m = m.numpy().astype(bool)
                vis[m] = (0.5 * vis[m] + 0.5 * col).astype(np.uint8)
        print(f"SAM3 CPU {time.time() - t0:.1f}s")
cv2.imwrite(a.out_png, vis)
print(f"peak RSS {PEAK[0]:.2f} GB -> {a.out_png}")
