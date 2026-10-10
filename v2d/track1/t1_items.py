"""Episode lists for the Track 1 pipeline (no heavy imports; usable from every env).

items(split) -> list of dicts: split, episode, sequence_id, video, T (video frames), frames (scored frames, sorted),
f0 (first scored frame), camera, object (id), noun (object id as words), prompt (object sentence).
  val    : FORM-HOI val set (/mnt/secondary/v2d/t1/formhoi_val; kitval/val_index.json, selection.json,
           hoi_metadata.yaml).  Video = the selected exo camera.
  track1 : dataset revision 5f68335 track_1 (meta/episodes_metadata.jsonl, meta/episodes.jsonl) + the kit sample
           submission (scored frames).  Cached in /mnt/secondary/v2d/t1/track1_index.json.
Integrity: the val list asserts that no val sequence is a Track 1 or Track 2 sequence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DATASET = Path(os.environ.get("V2D_DATASET", "/home/asubuntudesktop/TestingGrounds/egony/video_to_data_challenge"))
VAL_ROOT = Path("/mnt/secondary/v2d/t1/formhoi_val")
KIT = Path("/mnt/secondary/v2d/kit/v2d_submission_kit")
T1_INDEX = Path("/mnt/secondary/v2d/t1/track1_index.json")
WORK = Path("/mnt/secondary/v2d/t1")


def _blacklist():
    ids = set()
    for p in (DATASET / "track_1" / "meta" / "episodes_metadata.jsonl",
              DATASET / "track_2" / "tier_1_multiview_caption" / "meta" / "episodes_metadata.jsonl",
              DATASET / "track_2" / "tier_2_synthetic_noise" / "meta" / "episodes_metadata.jsonl"):
        if p.exists():
            for line in open(p):
                ids.add(json.loads(line)["sequence_id"])
    sel = json.load(open(VAL_ROOT / "selection.json"))
    ids |= set(sel.get("excluded_track2_ids", []))
    return ids


def _val_items():
    import yaml
    idx = {e["episode"]: e for e in json.load(open(VAL_ROOT / "kitval" / "val_index.json"))}
    sel = {int(r["val_index"]): r for r in json.load(open(VAL_ROOT / "selection.json"))["sequences"]}
    bl = _blacklist()
    out = []
    for ep in sorted(idx):
        e, r = idx[ep], sel[ep]
        assert e["sequence_id"] not in bl, f"val sequence {e['sequence_id']} is a Track 1/2 sequence"
        d = VAL_ROOT / e["sequence_id"]
        meta = yaml.safe_load(open(d / "hoi_metadata.yaml"))
        out.append({"split": "val", "episode": ep, "sequence_id": e["sequence_id"], "camera": e["camera"],
                    "video": str(d / f"videos__{e['camera']}.mp4"), "T": int(e["T"]), "frames": e["frames"],
                    "f0": int(e["frames"][0]), "object": str(meta["object"]["id"]),
                    "noun": str(meta["object"]["id"]).replace("_", " "),
                    "prompt": str(meta["object"]["prompt"]).rstrip(". ")})
    return out


def _track1_items():
    if not T1_INDEX.exists():
        import numpy as np
        import pandas as pd
        ids = pd.read_parquet(KIT / "data" / "track_1_sample_submission.parquet", columns=["row_id"])["row_id"].astype(str)
        p = ids.str.split("_", expand=True)
        ep, fr = p[1].astype(int), p[2].astype(int)
        keep = fr != 999999
        frames = {int(e): sorted(int(x) for x in np.unique(g["f"])) for e, g in
                  pd.DataFrame({"e": ep[keep], "f": fr[keep]}).groupby("e")}
        t1 = DATASET / "track_1"
        lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in open(t1 / "meta" / "episodes.jsonl")}
        out = []
        for line in open(t1 / "meta" / "episodes_metadata.jsonl"):
            m = json.loads(line)
            e = int(m["episode_index"])
            out.append({"split": "track1", "episode": e, "sequence_id": m["sequence_id"], "camera": m["camera"],
                        "video": str(t1 / "videos" / "chunk-000" / m["video_key"] / f"episode_{e:06d}.mp4"),
                        "T": int(lengths[e]), "frames": frames[e], "f0": int(frames[e][0]), "object": m["object"],
                        "noun": m["object"].replace("_", " "), "prompt": m["object_prompt"].rstrip(". ")})
        tmp = str(T1_INDEX) + ".tmp"
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, T1_INDEX)
    return json.load(open(T1_INDEX))


def items(split):
    return _val_items() if split == "val" else _track1_items()


def item(split, ep):
    return next(i for i in items(split) if i["episode"] == int(ep))


def window(it, pre=60, post=30):
    """[start, end) video frames to process: scored span plus margins (pre-contact frames help registration)."""
    return max(0, it["frames"][0] - pre), min(it["T"], it["frames"][-1] + 1 + post)


def ep_dir(kind, split, ep):
    return WORK / kind / split / f"episode_{int(ep):06d}"
