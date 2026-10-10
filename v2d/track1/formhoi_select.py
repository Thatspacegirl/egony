#!/usr/bin/env python3
"""Pick a FORM-HOI validation set for Track 1 local scoring (deterministic).

Rules enforced here (asserted, not just preferred):
  * none of the 30 Track 1 sequence ids, none of the Track 2 sequence ids (30 listed in tier_1 meta);
  * no sequence whose name contains a Track 1 object token (hula_hoop, big_red_bowl, iron, white_desk,
    black_pan, foam_grass_block, paint_roller, pink_foam_roll, short_wood_stool, white_laptop_cart);
  * objects are also chosen outside the Track 2 object list, so the set is clean for every track.
One sequence per object; camera assigned round-robin in Track 1 proportions (left 11, front 9, back 5, right 5).

usage: formhoi_select.py --archives formhoi_meta/manifest/archives.parquet --t1-meta .../track_1/meta/episodes_metadata.jsonl
                         --t2-meta .../track_2/tier_1_multiview_caption/meta/episodes_metadata.jsonl --n 25 --out selection.json
"""
import argparse, json, re
import numpy as np
import pandas as pd

T1_OBJECT_TOKENS = ("hula_hoop", "big_red_bowl", "iron", "white_desk", "black_pan", "foam_grass_block",
                    "paint_roller", "pink_foam_roll", "short_wood_stool", "white_laptop_cart")
# Objects with FORM-HOI sequences, outside both Track 1 and Track 2 object lists, roughly covering the
# Track 1 shape classes (small handheld, stick, ball, box/bin, furniture-sized, rolling).
CANDIDATE_OBJECTS = ("hatchet", "tennis_racket", "electric_drill_toy", "soup_can", "corn_can", "ceramic_vase",
                     "yoga_block", "dumbbell_15lb", "basketball", "yoga_ball", "soccer_ball", "wooden_crate",
                     "giant_cardboard_box", "g1_box", "keyboard_box", "beige_bin", "guitar", "microwave",
                     "air_purifier", "toy_car", "toy_airplane", "gray_case", "big_fish_hook", "wooden_sword",
                     "dark_blue_book", "disinfectant_wipes", "potato_light", "yellow_toy", "black_square_box",
                     "hand_soap", "tennis_ball", "luggage")
CAMERAS = ["left_stereo_camera_left"] * 11 + ["front_stereo_camera_left"] * 9 + \
          ["back_stereo_camera_left"] * 5 + ["right_stereo_camera_left"] * 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", required=True)
    ap.add_argument("--t1-meta", required=True)
    ap.add_argument("--t2-meta", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--exclude", default="", help="comma list of sequence ids to skip (e.g. unusable after fetch)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arch = pd.read_parquet(a.archives)
    t1 = [json.loads(l) for l in open(a.t1_meta)]
    t2 = [json.loads(l) for l in open(a.t2_meta)]
    t1_ids = {d["sequence_id"] for d in t1}
    t2_ids = {d["sequence_id"] for d in t2}
    t2_objs = {d["object"] for d in t2}
    skip = set(filter(None, a.exclude.split(",")))
    assert len(t1_ids) == 30 and len(t2_ids) >= 27, (len(t1_ids), len(t2_ids))
    rng = np.random.default_rng(a.seed)
    picks = []
    for obj in CANDIDATE_OBJECTS:
        assert obj not in t2_objs and not any(t in obj for t in T1_OBJECT_TOKENS), obj
        pat = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_" + re.escape(obj) + r"_[a-z0-9_]*\d+$")
        cands = sorted(s for s in arch.sequence_id if pat.match(s) and s not in skip
                       and not any(t in s for t in T1_OBJECT_TOKENS) and s not in t1_ids and s not in t2_ids)
        # avoid prefix collisions (e.g. 'luggage' vs 'rolling_luggage', 'tennis_ball' vs 'tennis_racket')
        if not cands:
            continue
        picks.append((obj, cands[int(rng.integers(len(cands)))], len(cands)))
        if len(picks) == a.n:
            break
    out = []
    cams = list(np.array(CAMERAS)[rng.permutation(len(CAMERAS))])
    for i, (obj, seq, ncand) in enumerate(picks):
        assert seq not in t1_ids and seq not in t2_ids
        assert not any(t in seq for t in T1_OBJECT_TOKENS)
        out.append({"val_index": i, "sequence_id": seq, "object": obj, "camera": str(cams[i % len(cams)]),
                    "n_candidates_for_object": ncand})
    json.dump({"seed": a.seed, "excluded_track1_ids": sorted(t1_ids), "excluded_track2_ids": sorted(t2_ids),
               "excluded_object_tokens": T1_OBJECT_TOKENS, "sequences": out}, open(a.out, "w"), indent=1)
    for r in out:
        print(r["val_index"], r["sequence_id"], r["camera"])
    print(f"{len(out)} sequences; Track1-id overlap: 0 (asserted); Track2-id overlap: 0 (asserted)")


if __name__ == "__main__":
    main()
