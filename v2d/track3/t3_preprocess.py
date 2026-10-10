"""Track 3 CPU preprocessing: decode the three ego streams, rectify the mono stereo pair, undistort cam_a.

Per episode writes  OUT/<split>/episode_XXXXXX/
    left/000000.jpg    rectified ego_cam_c (stereo LEFT), 1280x800 gray, K = K_rect
    right/000000.jpg   rectified ego_cam_b (stereo RIGHT), 1280x800 gray, same K, baseline ~0.0755 m
    cam_a/000000.jpg   ego_cam_a colour, undistorted to a square-pixel pinhole K = Ka_undist (2028x1520)
    meta.json          calibration (K_rect, baseline, Ka_undist, T_a<-rect, T_a<-c, ...), frame counts,
                       timestamps, object roster, rectification-quality stats
    fs_calibration.json  {fx, fy, cx, cy, baseline} for v2d_foundation_stereo run_image_list_to_depth
    cam_K.txt          3x3 K of cam_a undistorted (FoundationPose convention)
    left_K.txt         3x3 K of the rectified left camera
and OUT/preprocess_report.json (all episodes) + OUT/ego_calibration.json.

Usage (kit venv has cv2/pyarrow):
  OMP_NUM_THREADS=2 nice -n 10 python -I t3_preprocess.py --ds DATASET/track_3 --kit KIT \
      --out /mnt/secondary/v2d/t3/frames [--split public evaluation] [--episodes 12 ...] [--workers 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_calib import CAM_A_SIZE, STEREO_SIZE, calib_to_jsonable, load_ego_calibration, rectify_maps  # noqa: E402

FPS_NOMINAL = 20.25411910142744
STREAMS = {"a": "observation.images.ego_cam_a", "b": "observation.images.ego_cam_b",
           "c": "observation.images.ego_cam_c"}


def ffprobe_frames(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_frames,nb_read_packets", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["nb_frames"]), int(s["nb_read_packets"])


class Reader:
    """Sequential raw-frame reader through an ffmpeg pipe."""

    def __init__(self, path, w, h, gray):
        self.w, self.h, self.c = w, h, (1 if gray else 3)
        self.n = w * h * self.c
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-f", "rawvideo",
                                   "-pix_fmt", "gray" if gray else "bgr24", "-"],
                                  stdout=subprocess.PIPE, bufsize=self.n * 2)

    def read(self):
        buf = self.p.stdout.read(self.n)
        if len(buf) < self.n:
            return None
        a = np.frombuffer(buf, np.uint8)
        return a.reshape(self.h, self.w) if self.c == 1 else a.reshape(self.h, self.w, 3)

    def close(self):
        self.p.stdout.close()
        self.p.wait()
        return self.p.returncode


_sift = None


def rect_quality(L, R):
    """SIFT matches on the rectified pair -> vertical residual |dy| and disparity stats."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(3000)
    kl, dl = _sift.detectAndCompute(L, None)
    kr, dr = _sift.detectAndCompute(R, None)
    if dl is None or dr is None or len(kl) < 20 or len(kr) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.7 * b.distance]
    if len(good) < 20:
        return None
    pl = np.float64([kl[g.queryIdx].pt for g in good])
    pr = np.float64([kr[g.trainIdx].pt for g in good])
    dy = pl[:, 1] - pr[:, 1]
    disp = pl[:, 0] - pr[:, 0]
    inl = np.abs(dy) < 10  # drop gross mismatches before the stats
    return dict(n=int(len(good)), n_inl=int(inl.sum()), dy=dy[inl].astype(np.float32), x=pl[inl, 0].astype(np.float32),
                y=pl[inl, 1].astype(np.float32), disp=disp[inl].astype(np.float32))


def fit_dy_model(dy, x, y):
    """Robust LSQ fit dy ~ a + b*(x-640)/1000 + c*(y-400)/1000 (left-image coords)."""
    A = np.stack([np.ones_like(x), (x - 640) / 1000, (y - 400) / 1000], 1).astype(np.float64)
    d = dy.astype(np.float64)
    keep = np.ones(len(d), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        res = d - A @ coef
        mad = np.median(np.abs(res[keep])) + 1e-6
        keep = np.abs(res) < 4 * 1.4826 * mad
    return coef, res, keep


def dy_correction_pass(vid_b, vid_c, maps_l, maps_r, step, max_frames=40):
    """Pre-pass over the mono streams: per-episode systematic vertical residual of the factory rectification."""
    rb = Reader(vid_b, *STEREO_SIZE, gray=True)
    rc = Reader(vid_c, *STEREO_SIZE, gray=True)
    dys, xs, ys, i, used = [], [], [], 0, []
    while len(used) < max_frames:
        B, Cc = rb.read(), rc.read()
        if B is None or Cc is None:
            break
        if i % step == 0:
            s = rect_quality(cv2.remap(Cc, *maps_l, cv2.INTER_LINEAR), cv2.remap(B, *maps_r, cv2.INTER_LINEAR))
            if s is not None:
                dys.append(s["dy"]); xs.append(s["x"]); ys.append(s["y"]); used.append(i)
        i += 1
    rb.p.kill(); rc.p.kill(); rb.close(); rc.close()
    dy, x, y = np.concatenate(dys), np.concatenate(xs), np.concatenate(ys)
    coef, res, keep = fit_dy_model(dy, x, y)
    stats = dict(frames=used, n_matches=int(len(dy)), median_abs_dy_before=float(np.median(np.abs(dy))),
                 mean_dy_before=float(np.mean(dy)), a_px=float(coef[0]), b_px_per_kpx=float(coef[1]),
                 c_px_per_kpx=float(coef[2]), inlier_frac=float(keep.mean()),
                 median_abs_residual_fit=float(np.median(np.abs(res))))
    return coef, stats


def corrected_right_map(cal, coef):
    """map_new(x, y) = map_old(x, y - dy(x, y)) so the right image lines up with the left rows."""
    m1, m2 = cv2.initUndistortRectifyMap(cal["Kb"], cal["Db"], cal["R2"], cal["P2"], STEREO_SIZE, cv2.CV_32FC1)
    X, Y = np.meshgrid(np.arange(STEREO_SIZE[0], dtype=np.float32), np.arange(STEREO_SIZE[1], dtype=np.float32))
    dy = coef[0] + coef[1] * (X - 640) / 1000 + coef[2] * (Y - 400) / 1000
    Ys = (Y - dy).astype(np.float32)
    n1 = cv2.remap(m1, X, Ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    n2 = cv2.remap(m2, X, Ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.convertMaps(n1, n2, cv2.CV_16SC2)


def load_public_rows(ds, ep):
    import pyarrow.parquet as pq
    t = pq.read_table(f"{ds}/public/data/chunk-000/episode_{ep:06d}.parquet",
                      columns=["frame_index", "timestamp", "capture_time", "observation.objects"])
    fi = t.column("frame_index").to_numpy()
    ts = t.column("timestamp").to_numpy()
    ct = t.column("capture_time").to_numpy()
    objs = t.column("observation.objects")[0].as_py()
    return dict(n_rows=len(fi), frame_index_ok=bool((fi == np.arange(len(fi))).all()),
                timestamp=ts.tolist(), capture_time=ct.tolist(), objects=[o["name"] for o in objs])


def eval_roster(kit):
    j = json.load(open(f"{kit}/data/track_3_evaluation_objects.json"))
    return {int(k): v for k, v in j["episodes"].items()}


def eval_sample_frames(kit):
    import pyarrow.parquet as pq
    rid = pq.read_table(f"{kit}/data/track_3_sample_submission.parquet", columns=["row_id"]).column(0).to_pylist()
    nf = {}
    for r in rid:
        _, e, w, f, o = r.split("/")
        e, f = int(e[1:]), int(f[1:])
        nf[e] = max(nf.get(e, 0), f + 1)
    return nf


def process_episode(job):
    split, ep, args = job["split"], job["ep"], job["args"]
    cv2.setNumThreads(2)
    t0 = time.time()
    out_dir = f"{args['out']}/{split}/episode_{ep:06d}"
    meta_path = f"{out_dir}/meta.json"
    if os.path.exists(meta_path) and not args["force"]:
        return json.load(open(meta_path))["report"]
    cal = load_ego_calibration(args["calib"])
    (l1, l2), (r1, r2), (a1, a2) = rectify_maps(cal)
    vids = {k: f"{args['ds']}/{split}/videos/chunk-000/{v}/episode_{ep:06d}.mp4" for k, v in STREAMS.items()}
    probe = {k: ffprobe_frames(p) for k, p in vids.items()}
    dy_corr = None
    if args["refine_dy"]:
        coef, dy_corr = dy_correction_pass(vids["b"], vids["c"], (l1, l2), (r1, r2), args["refine_step"])
        r1, r2 = corrected_right_map(cal, coef)
    for sub in ("left", "right", "cam_a"):
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
    q = [cv2.IMWRITE_JPEG_QUALITY, int(args["jpeg_quality"])]
    ra = Reader(vids["a"], *CAM_A_SIZE, gray=False)
    rb = Reader(vids["b"], *STEREO_SIZE, gray=True)
    rc = Reader(vids["c"], *STEREO_SIZE, gray=True)
    n = 0
    counts = {"a": 0, "b": 0, "c": 0}
    checks = []
    while True:
        A, B, Cc = ra.read(), rb.read(), rc.read()
        for k, x in (("a", A), ("b", B), ("c", Cc)):
            counts[k] += x is not None
        if A is None or B is None or Cc is None:
            break
        L = cv2.remap(Cc, l1, l2, cv2.INTER_LINEAR)
        R = cv2.remap(B, r1, r2, cv2.INTER_LINEAR)
        Au = cv2.remap(A, a1, a2, cv2.INTER_LINEAR)
        cv2.imwrite(f"{out_dir}/left/{n:06d}.jpg", L, q)
        cv2.imwrite(f"{out_dir}/right/{n:06d}.jpg", R, q)
        cv2.imwrite(f"{out_dir}/cam_a/{n:06d}.jpg", Au, q)
        if n % args["check_every"] == 0:
            s = rect_quality(L, R)
            if s is not None:
                s["frame"] = n
                checks.append(s)
        n += 1
    # drain whatever is left so we count every frame of every stream
    for k, r in (("a", ra), ("b", rb), ("c", rc)):
        while r.read() is not None:
            counts[k] += 1
        r.close()

    dy = np.concatenate([c["dy"] for c in checks]) if checks else np.zeros(0)
    x = np.concatenate([c["x"] for c in checks]) if checks else np.zeros(0)
    y = np.concatenate([c["y"] for c in checks]) if checks else np.zeros(0)
    disp = np.concatenate([c["disp"] for c in checks]) if checks else np.zeros(0)
    # systematic residual model dy ~ a + b*(x-cx)/1000 + c*(y-cy)/1000 (pitch / roll / scale-like terms)
    fit = None
    if len(dy) > 50:
        Amat = np.stack([np.ones_like(x), (x - 640) / 1000, (y - 400) / 1000], 1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(Amat, dy.astype(np.float64), rcond=None)
        res = dy - Amat @ coef
        fit = dict(a_px=float(coef[0]), b_px_per_kpx=float(coef[1]), c_px_per_kpx=float(coef[2]),
                   median_abs_residual_after_fit=float(np.median(np.abs(res))))
    rect = dict(frames_checked=[c["frame"] for c in checks], n_matches=int(len(dy)),
                median_abs_dy=float(np.median(np.abs(dy))) if len(dy) else None,
                mean_dy=float(np.mean(dy)) if len(dy) else None,
                p90_abs_dy=float(np.percentile(np.abs(dy), 90)) if len(dy) else None,
                frac_abs_dy_lt_1px=float(np.mean(np.abs(dy) < 1)) if len(dy) else None,
                frac_disp_pos=float(np.mean(disp > 0)) if len(dy) else None,
                per_frame_median_abs_dy=[float(np.median(np.abs(c["dy"]))) for c in checks],
                linear_fit=fit)

    nb = {k: probe[k][0] for k in probe}
    report = dict(split=split, episode=ep, frames_written=n, decoded=counts, ffprobe_nb_frames=nb,
                  ffprobe_packets={k: probe[k][1] for k in probe}, rect=rect, right_dy_correction=dy_corr,
                  seconds=round(time.time() - t0, 1))
    ok = counts["a"] == counts["b"] == counts["c"] == n and all(nb[k] == n for k in nb)
    if split == "public":
        rows = load_public_rows(args["ds"], ep)
        report["parquet_rows"] = rows["n_rows"]
        ok = ok and rows["n_rows"] == n and rows["frame_index_ok"]
        timestamps, objects = rows["timestamp"], rows["objects"]
        report["capture_time_0"] = rows["capture_time"][0]
    else:
        timestamps = (np.arange(n) / FPS_NOMINAL).tolist()
        objects = args["eval_roster"][ep]
        report["sample_submission_frames"] = args["eval_frames"].get(ep)
        ok = ok and args["eval_frames"].get(ep) == n
    report["frame_count_ok"] = bool(ok)

    calj = calib_to_jsonable(cal)
    K_rect, Ka_u = cal["K_rect"], cal["Ka_undist"]
    meta = dict(
        split=split, episode=ep, n_frames=n, fps_nominal=FPS_NOMINAL, timestamps=timestamps,
        objects=objects, object_slots={i: o for i, o in enumerate(objects)},
        sources={k: os.path.relpath(v, args["ds"]) for k, v in vids.items()},
        dataset_revision="5f68335f3acc802033d1e80728c1633197521de8",
        images=dict(
            left=dict(dir="left", camera="ego_cam_c rectified (stereo LEFT)", size_wh=list(STEREO_SIZE),
                      K=K_rect.tolist(), gray=True),
            right=dict(dir="right", camera="ego_cam_b rectified (stereo RIGHT)", size_wh=list(STEREO_SIZE),
                       K=cal["P2"][:3, :3].tolist(), gray=True,
                       dy_correction=None if dy_corr is None else dict(
                           model="right_new(x,y) = right_factory(x, y - (a + b*(x-640)/1000 + c*(y-400)/1000))",
                           a_px=dy_corr["a_px"], b_px_per_kpx=dy_corr["b_px_per_kpx"],
                           c_px_per_kpx=dy_corr["c_px_per_kpx"])),
            cam_a=dict(dir="cam_a", camera="ego_cam_a undistorted pinhole", size_wh=list(CAM_A_SIZE),
                       K=Ka_u.tolist(), gray=False),
            jpeg_quality=int(args["jpeg_quality"]), filename="{frame:06d}.jpg"),
        stereo=dict(baseline_m=cal["baseline"], fx=float(K_rect[0, 0]), fy=float(K_rect[1, 1]),
                    cx=float(K_rect[0, 2]), cy=float(K_rect[1, 2]),
                    depth_from_disparity="depth_m = fx * baseline_m / disparity_px (left image grid)"),
        transforms={
            "T_a_rect": calj["T_a_rect"], "T_a_c": calj["T_a_c"], "T_rect_c": calj["T_rect_c"],
            "T_c_rect": calj["T_c_rect"], "T_b_c": calj["T_b_c"],
            "_convention": "T_dst_src: X_dst = T @ X_src (metres). rect = rectified-left camera frame "
                           "(frame of left/ depth), a = ego_cam_a (cam_a/ undistorted uses the same frame), "
                           "c = raw ego_cam_c, b = raw ego_cam_b."},
        calibration_full=calj,
        report=report,
    )
    json.dump(meta, open(meta_path + ".tmp", "w"), indent=1)
    os.replace(meta_path + ".tmp", meta_path)
    json.dump(dict(fx=float(K_rect[0, 0]), fy=float(K_rect[1, 1]), cx=float(K_rect[0, 2]), cy=float(K_rect[1, 2]),
                   baseline=cal["baseline"]), open(f"{out_dir}/fs_calibration.json", "w"), indent=1)
    write_v2d_intrinsics(out_dir, K_rect, Ka_u)
    np.savetxt(f"{out_dir}/cam_K.txt", Ka_u)
    np.savetxt(f"{out_dir}/left_K.txt", K_rect)
    pre = dy_corr["median_abs_dy_before"] if dy_corr else float("nan")
    print(f"[{split} ep{ep:3d}] frames={n} ok={ok} med|dy| factory={pre:.3f} -> written={rect['median_abs_dy']:.3f} "
          f"t={report['seconds']}s",
          flush=True)
    return report


def write_v2d_intrinsics(out_dir, K_rect, Ka_u):
    """v2d_common CameraIntrinsics JSONs (what run_video_to_poses / run_reconstruct --intrinsics expect)."""
    for name, K, (w, h) in (("cam_a_intrinsics.json", Ka_u, CAM_A_SIZE), ("left_intrinsics.json", K_rect, STEREO_SIZE)):
        json.dump(dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=int(w),
                       height=int(h)), open(f"{out_dir}/{name}", "w"), indent=1)


def safe(job):
    try:
        return process_episode(job)
    except Exception:
        return dict(split=job["split"], episode=job["ep"], error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="dataset track_3 dir (revision 5f68335)")
    ap.add_argument("--kit", required=True, help="submission kit root (eval roster + sample submission)")
    ap.add_argument("--out", default="/mnt/secondary/v2d/t3/frames")
    ap.add_argument("--split", nargs="*", default=["public", "evaluation"])
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--check_every", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_refine_dy", action="store_true",
                    help="disable the per-episode vertical-residual correction of the right rectification map")
    ap.add_argument("--refine_step", type=int, default=10)
    a = ap.parse_args()
    calib = f"{a.ds}/public/meta/camera_calibration.json"  # one ego device; eval ships no meta/
    args = dict(ds=a.ds, out=a.out, calib=calib, jpeg_quality=a.jpeg_quality, check_every=a.check_every,
                force=a.force, refine_dy=not a.no_refine_dy, refine_step=a.refine_step, eval_roster=eval_roster(a.kit), eval_frames=eval_sample_frames(a.kit))
    os.makedirs(a.out, exist_ok=True)
    json.dump(calib_to_jsonable(load_ego_calibration(calib)), open(f"{a.out}/ego_calibration.json", "w"), indent=1)
    jobs = []
    for split in a.split:
        eps = sorted(int(os.path.basename(p)[8:14]) for p in
                     glob.glob(f"{a.ds}/{split}/videos/chunk-000/{STREAMS['a']}/episode_*.mp4"))
        if a.episodes:
            eps = [e for e in eps if e in a.episodes]
        jobs += [dict(split=split, ep=e, args=args) for e in eps]
    print(f"{len(jobs)} episodes, {a.workers} workers", flush=True)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        reports = list(pool.imap_unordered(safe, jobs))
    reports.sort(key=lambda r: (r["split"], r["episode"]))
    rp = f"{a.out}/preprocess_report.json"
    old = json.load(open(rp)) if os.path.exists(rp) else []
    keep = {(r["split"], r["episode"]): r for r in old}
    keep.update({(r["split"], r["episode"]): r for r in reports})
    allr = sorted(keep.values(), key=lambda r: (r["split"], r["episode"]))
    json.dump(allr, open(rp, "w"), indent=1)
    errs = [r for r in reports if "error" in r]
    bad = [r for r in reports if "error" not in r and not r["frame_count_ok"]]
    for r in errs:
        print("ERROR", r["split"], r["episode"], r["error"])
    print(f"done: {len(reports) - len(errs)} ok, {len(errs)} errors, {len(bad)} frame-count mismatches")


if __name__ == "__main__":
    main()
