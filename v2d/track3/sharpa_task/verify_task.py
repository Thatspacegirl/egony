"""Verify a generated Track 3 floating-Sharpa task against FlashCHORD (CPU only).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= /mnt/secondary/v2d/envs/flash_chord/bin/python verify_task.py \
        MANIFEST.json [--scene] [--no-hydra]

Checks (exit code 1 if any FAIL):
  * flash_chord.data.load_reference(task, control_fps=20, motion_speed=1.0) -> exactly N frames, not resampled
    (and shows what the recipe's motion_speed=0.5 would have produced),
  * object_body_names == manifest roster order; one rigid URDF per body and each resolves to our file,
  * support USDA resolves through FlashCHORD's own lookup and parses (Cube top == table height),
  * frame-0: every object's lowest vertex is on/above the support top (within tolerance) and inside it,
  * harness episode regex on the task path == episode index,
  * Hydra composes train_flash_sac.py with the task (canonical recipe + motion_speed=1.0, and the 3090 preset),
  * --scene: builds the real Newton training scene on CPU (SharpaHands + objects with CoACD + support,
    world_count=1) and checks spawned object poses == reference frame 0 and the imported mass/COM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
V2D_FC = Path("/mnt/secondary/v2d/video_to_data/robotic_grounding/flash_chord")
FLASH_PY = "/mnt/secondary/v2d/envs/flash_chord/bin/python"
PRESET_DIR = HERE / "configs"
GAP_TOL = 1.0e-3  # m below the support top tolerated at frame 0

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def hydra_compose(parquet: str, extra: list[str], preset: bool) -> dict:
    cmd = [FLASH_PY, str(V2D_FC / "scripts" / "train_flash_sac.py")]
    if preset:
        cmd += ["--config-dir", str(PRESET_DIR), "experiment=sharpa_flash_sac_3090"]
    else:
        cmd += ["experiment=sharpa_flash_sac", "task.motion_speed=1.0"]
    cmd += [f"task.parquet='{parquet}'", *extra, "--cfg", "job", "--resolve"]
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
        leftovers = sorted(p.name for p in Path(cwd).iterdir())
    if r.returncode != 0:
        raise RuntimeError(f"hydra compose failed:\n{r.stderr[-3000:]}")
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.create(r.stdout), resolve=False)
    cfg["_cmd"] = " ".join(cmd)
    cfg["_cwd_leftovers"] = leftovers
    return cfg


def parse_support(usda: Path) -> list[dict]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usda))
    out = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Cube":
            continue
        t = np.array(prim.GetAttribute("xformOp:translate").Get(), float)
        s = np.array(prim.GetAttribute("xformOp:scale").Get(), float)
        e = float(prim.GetAttribute("size").Get())
        dims = e * np.abs(s)
        out.append({"center": t, "dims": dims, "top": t[2] + 0.5 * dims[2]})
    return out


def urdf_mesh(urdf: Path):
    import trimesh

    root = ET.parse(urdf).getroot()
    vis = root.find("link/visual/geometry/mesh")
    col = root.find("link/collision/geometry/mesh")
    if vis is None or col is None or vis.get("filename") != col.get("filename"):
        raise ValueError(f"{urdf}: visual and collision must reference the same mesh")
    path = (urdf.parent / vis.get("filename")).resolve()
    inert = root.find("link/inertial")
    com = np.array([float(x) for x in inert.find("origin").get("xyz").split()])
    mass = float(inert.find("mass").get("value"))
    return path, trimesh.load(path, force="mesh", process=False), mass, com


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--scene", action="store_true")
    ap.add_argument("--no-hydra", action="store_true")
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    N, names, ep = man["num_frames"], man["object_body_names"], man["episode_index"]
    task = man["task_parquet"]
    print(f"== verify {man['sequence_id']}  N={N}  objects={names}\n   task={task}")

    from flash_chord.assets.registry import support_usda_for_reference
    from flash_chord.data import load_reference
    from flash_chord.data.resampling import playback_times

    ref = load_reference(task, control_fps=20.0, motion_speed=1.0)
    md = ref.metadata
    check("load_reference(control_fps=20, motion_speed=1.0) frames == N", ref.num_frames == N,
          f"{ref.num_frames} vs {N}, source_fps={md.source_fps}, resampled={md.is_resampled}")
    check("reference not resampled at 20 Hz / speed 1.0", not md.is_resampled and md.source_fps == 20.0)
    half = len(playback_times(N, md.source_fps, 20.0, 0.5)[1])
    print(f"   (for contrast: the recipe default motion_speed=0.5 would give {half} rollout steps)")
    check("object_body_names == roster order", list(ref.object_body_names()) == names, str(ref.object_body_names()))
    assets = ref.object_assets()
    urdfs = [str(s.urdf_path) for s in assets]
    check("one rigid URDF per body (ours)", len(assets) == len(names)
          and all(u == o["urdf"] for u, o in zip(urdfs, man["objects"])), str(urdfs))
    m = re.search(r"episode_(\d+)", str(task))
    check("harness episode regex", bool(m) and int(m.group(1)) == ep, m.group(0) if m else "no match")

    sup = support_usda_for_reference(ref)
    check("support USDA resolves via FlashCHORD lookup", sup is not None and Path(sup) == Path(man["support"]["usda"]),
          str(sup))
    boxes = parse_support(Path(sup)) if sup else []
    top = max(bx["top"] for bx in boxes) if boxes else float("nan")
    check("support parses (Cube) and top == table z", len(boxes) == 1 and abs(top - man["support"]["top_z"]) < 1e-6,
          f"top={top:.4f}")

    pos0 = np.asarray(ref.object_body_pos_w())[0]
    quat0 = np.asarray(ref.object_body_quat_w())[0]
    from sharpa_task.checks import pair_penetration_report, support_gap_report

    meshes, verts = [], []
    for o in man["objects"]:
        path, mesh, mass, com = urdf_mesh(Path(o["urdf"]))
        check(f"{o['name']}: URDF mesh is the budgeted/submitted mesh", str(path) == o["obj"]
              and len(mesh.faces) <= 4096 and len(mesh.vertices) <= 4096, f"{len(mesh.vertices)}v/{len(mesh.faces)}f")
        check(f"{o['name']}: plausible mass, COM inside hull bbox", 0.03 <= mass <= 5.0
              and np.all(com >= mesh.bounds[0] - 1e-6) and np.all(com <= mesh.bounds[1] + 1e-6),
              f"{mass:.3f} kg, com={np.round(com, 4).tolist()}")
        meshes.append(mesh)
        verts.append(np.asarray(mesh.vertices))
    bx = boxes[0] if boxes else {"center": np.zeros(3), "dims": np.ones(3)}
    gaps = support_gap_report(verts, pos0, quat0, names, top, bx["center"][:2], bx["dims"][:2])
    settled = man.get("settle", {}).get("applied", {})
    for name, g in gaps.items():
        # settled objects rest on their CoACD hulls, which deviate from the mesh by ~1-3 mm
        tol = 0.003 if name in settled else GAP_TOL
        check(f"frame0 {name}: no support interpenetration", g["gap_m"] >= -tol and g["inside_footprint"],
              f"lowest vertex {g['gap_m']*1000:+.2f} mm vs support top (tol {tol*1000:.0f} mm"
              + (", settled on hulls)" if name in settled else ")"))
    for pair, r in pair_penetration_report(meshes, pos0, quat0, names).items():
        print(f"   [info] frame0 {pair}: min dist {r['min_dist_m']*1000:.1f} mm, "
              f"pseudo-penetration {r['max_penetration_m']*1000:.1f} mm")
    # placeholder / hand frames vs objects at frame 0 (info)
    for side in ("right", "left"):
        fr = np.asarray(ref.robot_frame_pos_w(side))[0] if hasattr(ref, "robot_frame_pos_w") else None
        if fr is not None:
            print(f"   [info] {side} hand frames z-range at frame0: {fr[:, 2].min():.3f}..{fr[:, 2].max():.3f}")

    if not a.no_hydra:
        for preset in (False, True):
            label = "3090 preset" if preset else "canonical recipe + motion_speed=1.0"
            try:
                cfg = hydra_compose(task, [], preset)
            except RuntimeError as exc:
                check(f"hydra compose ({label})", False, str(exc)[-500:])
                continue
            tr = cfg["training"]
            detail = (f"motion_speed={cfg['task']['motion_speed']} world_count={cfg['scene']['world_count']} "
                      f"upc={tr['updates_per_collection']} replay={tr['replay']['capacity']}/"
                      f"{tr['replay']['minimum_size']} obs={tr['replay']['observation_storage_dtype']} "
                      f"steps={tr['total_environment_steps']} ckpt={tr['checkpoint']['save_interval_environment_steps']}")
            ok = (cfg["task"]["motion_speed"] == 1.0 and cfg["task"]["parquet"] == task
                  and tr["total_environment_steps"] % cfg["scene"]["world_count"] == 0
                  and cfg["sim"]["fps"] == 20.0 and not cfg["_cwd_leftovers"])
            if preset:
                ok &= (cfg["scene"]["world_count"] == 2048 and tr["updates_per_collection"] == 4.0
                       and tr["replay"]["observation_storage_dtype"] == "float16")
            check(f"hydra compose ({label})", ok, detail)

    if a.scene:
        scene_check(task, man, ref)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    out = a.manifest.with_name(a.manifest.stem + "_verify.json")
    out.write_text(json.dumps([{"check": n, "ok": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
    return 1 if n_fail else 0


def scene_check(task: str, man: dict, ref) -> None:
    """Build the real training scene (world_count=1) on CPU and compare against the reference."""
    import time

    import warp as wp

    import newton
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    t0 = time.time()
    with wp.ScopedDevice("cpu"):
        setup = setup_scene(parquet=task, control_fps=20.0, motion_speed=1.0, embodiment=SharpaHands(),
                            collision=CollisionPolicy(robot_support_collision=False, ground=True),
                            world_count=1, include_support=True, decompose_objects=True)
        model = setup.scene.model
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        mass = model.body_mass.numpy()
        com = model.body_com.numpy()
        inertia = model.body_inertia.numpy()
    pos0 = np.asarray(ref.object_body_pos_w())[0]
    quat0 = np.asarray(ref.object_body_quat_w())[0]
    for k, binding in enumerate(setup.scene.objects):
        bid = binding.bodies[0].body_id
        o = man["objects"][k]
        dp = float(np.linalg.norm(body_q[bid][:3] - pos0[k]))
        q = body_q[bid][3:7]  # xyzw
        dq = float(abs(abs(np.dot(q, quat0[k][[1, 2, 3, 0]])) - 1.0))
        hulls = binding.shapes.stop - binding.shapes.start
        check(f"scene: {o['name']} spawned at reference frame 0", dp < 1e-4 and dq < 1e-5, f"|dp|={dp:.2e} |dq|={dq:.2e}")
        idiag = np.diag(np.asarray(inertia[bid]).reshape(3, 3))
        check(f"scene: {o['name']} mass/COM/inertia as authored", abs(mass[bid] - o["mass_kg"]) < 1e-4
              and np.allclose(com[bid], o["com"], atol=1e-5)
              and np.allclose(idiag, o["inertia_diag"], rtol=1e-3, atol=1e-9),
              f"mass={mass[bid]:.4f} I_diag={np.round(idiag, 6).tolist()} shapes(incl. hulls)={hulls}")
    check("scene: support built", setup.scene.support is not None, f"{setup.support_usda}")
    print(f"   [info] scene: bodies={model.body_count} shapes={model.shape_count} build {time.time()-t0:.1f}s on CPU")


if __name__ == "__main__":
    raise SystemExit(main())
