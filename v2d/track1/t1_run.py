#!/usr/bin/env python3
"""Track 1 end-to-end driver (CPU): build raw episode predictions from our stage outputs, then score (val).

  build --split val|track1 --out DIR [--variant s3gx] [--episodes 0-12] [--no-k]
        -> DIR/episode_%06d.npz + episode_%06d_object.ply + episode_%06d_assemble.json
  then post-process with postprocess.py (smoothing + PEN), e.g.
        postprocess.py --in DIR --out DIR_post --val-index /mnt/secondary/v2d/t1/formhoi_val/kitval/val_index.json
  score --pred DIR [--episodes ...] [--no-pen] [--json out.json]
        kit metric code (score_t1.FastScorer); tune split = val 0-12, held-out = 13-24; leaderboard rows for reference.

Python: /mnt/secondary/v2d/scratch/venv/bin/python -I ; OMP/MKL/NUMBA threads 4, nice -n 10.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LEADERBOARD = {"RoboPanda": (15.175, 22.087, 0.13902, 0.10591, 0.0017),
               "ByeByeHelloHaHaHa": (14.205, 31.406, 0.16957, 0.13563, 0.00058),
               "CARI4D baseline": (15.575, 23.162, 1.974, 0.726, 0.00423)}
KEYS = ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm", "interpenetration_cm")
SHORT = ("CD-H", "CD-O", "ACC-H", "ACC-O", "PEN")


def parse_eps(s, all_eps):
    if not s:
        return sorted(all_eps)
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [e for e in out if e in set(all_eps)]


def oracle_object(ep, epd):
    """VAL DIAGNOSTIC ONLY: replace our object by the GT object mapped through the scorer's own f0 Sim(3)
    (pred human -> GT human), i.e. the CD-O / ACC-O our HUMAN track alone would allow.  Reads GT."""
    import score_t1 as S
    import t1lib as L
    fs = S.FastScorer()
    gt, vf, ref = fs.ref(ep)
    frames = np.asarray(fs.index[ep]["frames"])
    pred, _, _ = fs.arrays(ep, epd, vf)
    s, Rm, t = fs.cdh.fit_similarity_transform(pred["mhr_vertices"][0], ref["mhr_vertices"][0])
    # p_gt = s * p_pred @ Rm + t  (row form)  ->  p_pred = (p_gt - t) @ Rm.T / s
    Rg, tg = np.asarray(gt["object_rotation"]), np.asarray(gt["object_translation"])
    T = len(epd["pose"])
    Tg = len(Rg)
    out = dict(epd)
    Ro = np.einsum("ij,tjk->tik", Rm, Rg[:min(T, Tg)])
    to = (tg[:min(T, Tg)] - t) @ Rm.T / s
    if T > Tg:
        Ro = np.concatenate([Ro, np.repeat(Ro[-1:], T - Tg, 0)]); to = np.concatenate([to, np.repeat(to[-1:], T - Tg, 0)])
    out["object_rotation"], out["object_translation"] = Ro, to
    out["object_scale"] = np.array(float(np.asarray(gt["object_scale"]).reshape(())) / s)
    return out, fs.val / "gt" / f"episode_{ep:06d}_object.glb"


def cmd_build(a):
    import t1_items as I
    import t1_assemble as A
    its = {it["episode"]: it for it in I.items(a.split)}
    eps = parse_eps(a.episodes, its)
    ok = []
    for ep in eps:
        t0 = time.time()
        try:
            if a.oracle_object:
                assert a.split == "val"
                import t1_human as HU
                _, s3p, gxp = HU.find_outputs(a.split, ep)
                hum, hinfo = HU.build_episode(s3p, gxp, a.variant)
                epd, mesh = oracle_object(ep, {**hum, **HU.dummy_object(len(hum["pose"]))})
                A.write(a.out, ep, epd, mesh, {"oracle_object": True, "human": hinfo, "object_scale": float(epd["object_scale"])})
                ok.append(ep); print(f"ep {ep:2d} oracle-object built", flush=True)
                continue
            epd, mesh, info = A.build(a.split, ep, variant=a.variant, use_k=not a.no_k)
            A.write(a.out, ep, epd, mesh, info)
            ok.append(ep)
            print(f"ep {ep:2d} built k={info.get('k', 1):.3f} scale={info['object_scale']:.3f} "
                  f"fpose {info['fpose_ok']}/{info['fpose_n']} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ep {ep:2d} FAILED: {e!r}", flush=True)
            traceback.print_exc()
    print("built", ok)


def cmd_score(a):
    import score_t1 as S
    import t1lib as L
    fs = S.FastScorer()
    pred = Path(a.pred)
    eps = parse_eps(a.episodes, fs.index)
    rows = {}
    for ep in eps:
        npz = pred / f"episode_{ep:06d}.npz"
        mesh = next((p for p in (pred / f"episode_{ep:06d}_object{s}" for s in (".ply", ".glb", ".obj")) if p.exists()), None)
        if not npz.exists() or mesh is None:
            continue
        d = L.load_episode(npz)
        m = fs.score_episode(ep, d, fs.budget(mesh), pen=not a.no_pen)
        rows[ep] = m
        print(f"ep {ep:2d} {fs.index[ep]['sequence_id'][20:60]:40s} " +
              " ".join(f"{s} {m.get(k, np.nan):8.4f}" for s, k in zip(SHORT, KEYS)), flush=True)
    res = {"per_episode": rows}
    for name, sel in (("tune 0-12", range(0, 13)), ("held-out 13-24", range(13, 25)), ("all", range(0, 25))):
        es = [e for e in sel if e in rows]
        if not es:
            continue
        mean = {k: float(np.mean([rows[e].get(k, np.nan) for e in es])) for k in KEYS}
        res[name] = {"n": len(es), **mean}
        print(f"{name:15s} n={len(es):2d} " + " ".join(f"{s} {mean[k]:8.4f}" for s, k in zip(SHORT, KEYS)))
    for name, v in LEADERBOARD.items():
        print(f"{name:15s} (Kaggle)  " + " ".join(f"{s} {x:8.4f}" for s, x in zip(SHORT, v)))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cdh_per_frame_aligned(fs, ep, epd, vf, stride=5):
    """Diagnostic: CD-H if EVERY frame got its own Sim(3) (articulation/shape error only, no trajectory error)."""
    _, _, ref = fs.ref(ep)
    pred, frames, _ = fs.arrays(ep, epd, vf)
    cdh = fs.cdh
    vals = []
    for i in range(0, len(frames), stride):
        tr = cdh.fit_similarity_transform(pred["mhr_vertices"][i], ref["mhr_vertices"][i])
        ph = cdh.apply_similarity_transform(pred["human_surface_points"][i:i + 1], tr)
        vals.append(cdh.symmetric_chamfer_distance(ph, ref["human_surface_points"][i:i + 1]) * 100.0)
    return float(np.mean(vals))


def cmd_eval(a):
    """In-memory tuning loop on val (no files): raw build (cached) -> [contact depth] -> smoothing -> score (no PEN)."""
    import pickle
    import score_t1 as S
    import t1_assemble as A
    import t1_hoi as HOI
    from smoothing import SmoothCfg, smooth_episode
    fs = S.FastScorer()
    eps = parse_eps(a.episodes, fs.index)
    cache = Path("/mnt/secondary/v2d/t1/work/cache"); cache.mkdir(parents=True, exist_ok=True)
    cfgs = {}
    for name in a.smooth.split(","):
        cfgs[name] = None if name == "none" else SmoothCfg(**json.load(open(HERE / f"smooth_{name}.json")))
    rows = []
    for ep in eps:
        for var in a.variants.split(","):
            cp = cache / f"val_{ep:02d}_{var}.pkl"
            try:
                if cp.exists() and not a.rebuild:
                    epd, mesh, info = pickle.load(open(cp, "rb"))
                else:
                    epd, mesh, info = A.build("val", ep, variant=var)
                    pickle.dump((epd, mesh, info), open(cp, "wb"))
            except Exception as e:  # noqa: BLE001
                print(f"ep {ep} {var}: build failed {e!r}"); continue
            vf = fs.budget(mesh)
            frames = fs.index[ep]["frames"]
            f0 = frames[0]
            for hoi in a.hoi.split(","):
                e1, hinfo = (epd, {})
                if hoi == "contact":
                    e1, hinfo = HOI.contact_depth(epd, vf, frames)
                for cname, cfg in cfgs.items():
                    e2 = smooth_episode(e1, f0, cfg) if cfg is not None else e1
                    m = fs.score_episode(ep, e2, vf, pen=False)
                    r = {"ep": ep, "variant": var, "hoi": hoi, "smooth": cname, **{k: m[k] for k in KEYS if k in m},
                         "k": info.get("k", np.nan)}
                    if a.pa:
                        r["cdh_pa"] = cdh_per_frame_aligned(fs, ep, e2, vf)
                    rows.append(r)
                    print(f"ep {ep:2d} {var:6s} {hoi:8s} {cname:10s} " + " ".join(f"{s_} {m.get(k, np.nan):8.3f}" for s_, k in zip(SHORT[:4], KEYS[:4])) +
                          f"  k={info.get('k', np.nan):.3f}" + (f" hoi={hinfo}" if hinfo and cname == list(cfgs)[0] else ""), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    if len(df):
        df["split"] = np.where(df.ep <= 12, "tune", "heldout")
        cols = [k for k in list(KEYS[:4]) + ["cdh_pa"] if k in df]
        print(df.groupby(["split", "variant", "hoi", "smooth"])[cols].mean().round(4).to_string())
    if a.json:
        df.to_json(a.json, orient="records", indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--split", required=True); b.add_argument("--out", required=True)
    b.add_argument("--variant", default="s3gx"); b.add_argument("--episodes", default="")
    b.add_argument("--no-k", action="store_true")
    b.add_argument("--oracle-object", action="store_true", help="VAL diagnostic: GT object through the f0 Sim(3)")
    s = sp.add_parser("score")
    s.add_argument("--pred", required=True); s.add_argument("--episodes", default="")
    s.add_argument("--no-pen", action="store_true"); s.add_argument("--json")
    e = sp.add_parser("eval")
    e.add_argument("--variants", default="mg"); e.add_argument("--episodes", default="")
    e.add_argument("--smooth", default="none,default"); e.add_argument("--hoi", default="none")
    e.add_argument("--rebuild", action="store_true"); e.add_argument("--json")
    e.add_argument("--pa", action="store_true", help="also per-frame-aligned CD-H (articulation only)")
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
