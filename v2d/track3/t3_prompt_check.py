"""GPU check of SAM3 text prompts (image mode, Sam3Model) on cam_a frames -> overlay sheets to LOOK at.

For each (split, episode) and a few frames, runs every candidate phrase of every roster object and draws all
instances above the threshold (colour per instance, label 'phrase score').  One JPEG per episode in OUT.

  /mnt/secondary/v2d/envs/t3-seg/bin/python -I t3_prompt_check.py --eps public:21 public:18 evaluation:3 \
      --out /mnt/secondary/v2d/t3/checks/prompts
"""
import argparse
import json
import os

import cv2
import numpy as np

CANDIDATES = {
    "white_pot": ["white pot", "white saucepan", "pot"],
    "white_pot_lid": ["pot lid", "lid"],
    "wooden_spoon": ["wooden spoon", "spoon"],
    "blue_cup": ["blue cup"],
    "beige_cup": ["beige cup", "cup"],
    "plastic_dish_rack": ["dish rack", "white dish rack"],
    "water_pitcher": ["watering can", "white pitcher", "white jug"],
    "mini_sweeper": ["hand brush", "brush", "small broom"],
    "mini_dust_pan": ["dustpan", "yellow dustpan"],
    "wooden_piece_1": ["wooden block", "wooden board"],
    "wooden_piece_2": ["wooden block", "wooden board"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="/mnt/secondary/v2d/t3/frames")
    ap.add_argument("--eps", nargs="+", required=True, help="split:episode")
    ap.add_argument("--frames", default="0,0.33,0.66", help="absolute ints or fractions of the clip")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/mnt/secondary/v2d/weights/sam3/sam3")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(a.weights)
    model = Sam3Model.from_pretrained(a.weights, torch_dtype=torch.bfloat16).to("cuda").eval()
    os.makedirs(a.out, exist_ok=True)
    report = {}
    for spec in a.eps:
        sp, e = spec.split(":")
        ep_dir = f"{a.frames_root}/{sp}/episode_{int(e):06d}"
        meta = json.load(open(f"{ep_dir}/meta.json"))
        n = meta["n_frames"]
        fr = []
        for x in a.frames.split(","):
            v = float(x)
            fr.append(int(v * (n - 1)) if (0 < v < 1) else int(v))
        phrases = list(dict.fromkeys(p for o in meta["objects"] for p in CANDIDATES[o])) + (["hand"] if a.hand else [])
        rows = []
        for f in fr:
            im = cv2.imread(f"{ep_dir}/cam_a/{f:06d}.jpg")
            im = cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            tiles = []
            for ph in phrases:
                inp = proc(images=pil, text=ph, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
                with torch.inference_mode():
                    out = model(**inp)
                r = proc.post_process_instance_segmentation(out, threshold=a.thr, mask_threshold=0.5,
                                                            target_sizes=inp.get("original_sizes").tolist())[0]
                vis = im.copy()
                sc = r["scores"].float().cpu().numpy()
                rep = []
                for k, m in enumerate(r["masks"]):
                    m = m.cpu().numpy().astype(bool)
                    col = np.array([(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 128, 0), (0, 0, 255)][k % 5], np.uint8)
                    vis[m] = (0.45 * vis[m] + 0.55 * col).astype(np.uint8)
                    ys, xs = np.nonzero(m)
                    if len(xs):
                        cv2.putText(vis, f"{k}:{sc[k]:.2f}", (int(xs.mean()), int(ys.mean())), 0, 0.8, (255, 255, 255), 2)
                        rep.append(dict(score=round(float(sc[k]), 3), px=int(m.sum()),
                                        bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
                cv2.putText(vis, f"{sp[:4]}{e} f{f} '{ph}' n={len(rep)}", (8, 30), 0, 0.9, (0, 255, 255), 2)
                tiles.append(cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
                report[f"{spec}/f{f}/{ph}"] = rep
            rows.append(np.hstack(tiles))
        W = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(f"{a.out}/{sp}_{int(e):02d}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(spec, "done", flush=True)
    json.dump(report, open(f"{a.out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
