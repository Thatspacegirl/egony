"""Frame-0 physical consistency with the SIM collision geometry (CPU MuJoCo, objects + support only).

    # report for a generated task (reads its manifest + task parquet)
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= /mnt/secondary/v2d/envs/flash_chord/bin/python settle_check.py MANIFEST.json
    # raw mode used by to_sharpa_task.py --settle-static-start (returns settled poses)
    ... settle_check.py --raw IN.json --out OUT.json

Builds a plain MuJoCo model with the support Cube, the z=0 ground plane and every object as a free body
(authored URDF mass/COM/inertia) whose collision geoms are the SAME CoACD hulls FlashCHORD trains with:
meshes are loaded with newton's own loader, so convex_decompose (training defaults, shared
FLASH_CHORD_CACHE_DIR) hits the cache written by the scene builder. Objects are placed at frame 0.
Reports
  * frame-0 contact penetration per body pair (negative = hulls overlap),
  * drift (position / rotation) after 0.25 s and the full horizon with no hands: what a resting object does
    at reset in the scored rollout (the evaluator's explicit reset starts objects at reference frame 0).
Approximation of FlashCHORD's Newton/MuJoCo-Warp scene: same contact model family, default solref/solimp,
friction 1.0, 100 Hz physics (sim/rl.yaml: 20 fps x 5 substeps), pyramidal cone, impratio 20. No hands.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def urdf_inertial(urdf: str):
    root = ET.parse(urdf).getroot()
    inert = root.find("link/inertial")
    com = [float(x) for x in inert.find("origin").get("xyz").split()]
    mass = float(inert.find("mass").get("value"))
    i = inert.find("inertia").attrib
    full = [float(i[k]) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    return mass, com, full


def support_boxes_from_usda(usda: str) -> list[tuple[list[float], list[float]]]:
    from pxr import Usd

    out = []
    stage = Usd.Stage.Open(usda)  # keep a reference: a temporary stage expires mid-traversal
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Cube":
            c = np.array(prim.GetAttribute("xformOp:translate").Get(), float)
            s = np.array(prim.GetAttribute("xformOp:scale").Get(), float) * float(prim.GetAttribute("size").Get())
            out.append((c.tolist(), s.tolist()))
    return out


def run_settle(objects, pos0, quat0, boxes, static=None, seconds=1.0, dt=0.01) -> dict:
    """objects: [{"name","obj","urdf"}]; pos0 (B,3); quat0 (B,4) wxyz; boxes: [(center, full_size)].

    static[b] = True keeps that body fixed (e.g. already held/moving at frame 0). Returns the report incl.
    settled poses ("final_pos", "final_wxyz") after ``seconds``.
    """
    import mujoco
    from flash_chord.assets.mesh import convex_decompose
    from newton._src.utils.mesh import load_meshes_from_file  # exactly what newton's URDF importer does

    import trimesh

    B = len(objects)
    static = [False] * B if static is None else list(static)
    fr = [1.0, 0.005, 0.0001]
    spec = mujoco.MjSpec()
    spec.option.timestep = dt
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 20.0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    wb = spec.worldbody
    wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], friction=fr)
    for i, (c, s) in enumerate(boxes):
        wb.add_geom(name=f"support{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.5 * np.asarray(s)).tolist(),
                    pos=list(c), friction=fr)
    hull_counts = {}
    for k, o in enumerate(objects):
        nm = load_meshes_from_file(o["obj"], scale=np.ones(3), maxhullvert=64)
        if len(nm) != 1:
            raise SystemExit(f"{o['obj']}: expected one mesh, got {len(nm)}")
        hulls = convex_decompose(np.asarray(nm[0].vertices, np.float64), np.asarray(nm[0].indices, np.int64).reshape(-1, 3))
        mass, com, full = urdf_inertial(o["urdf"])
        body = wb.add_body(name=o["name"], pos=list(map(float, pos0[k])), quat=list(map(float, quat0[k])))
        if not static[k]:
            body.add_freejoint()
        body.mass, body.ipos, body.fullinertia, body.explicitinertial = mass, com, full, True
        n = 0
        for j, h in enumerate(hulls):
            hv = np.asarray(h.vertices, float)
            if len(hv) < 4 or trimesh.Trimesh(hv).convex_hull.volume < 1e-10:
                continue
            name = f"{o['name']}_h{j}"
            spec.add_mesh(name=name, uservert=hv.ravel().tolist())
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, friction=fr)
            n += 1
        hull_counts[o["name"]] = n
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def owner(geom_id: int) -> str:
        b = model.geom_bodyid[geom_id]
        if b == 0:
            return "world:" + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)

    ncon0 = int(data.ncon)
    pen: dict[str, float] = {}
    for c in data.contact[: data.ncon]:
        key = "|".join(sorted((owner(c.geom1), owner(c.geom2))))
        pen[key] = min(pen.get(key, 0.0), float(c.dist))

    def pose(k):
        bid = model.body(objects[k]["name"]).id
        return data.xpos[bid].copy(), data.xquat[bid].copy()

    start = [pose(k) for k in range(B)]
    steps = int(round(seconds / dt))
    marks = {int(round(0.25 / dt)): "0.25s", steps: f"{seconds:g}s"}
    drift: dict = {}
    for i in range(1, steps + 1):
        mujoco.mj_step(model, data)
        if i in marks:
            for k, o in enumerate(objects):
                p, q = pose(k)
                dp = np.linalg.norm(p - start[k][0])
                ang = 2 * np.degrees(np.arccos(np.clip(abs(np.dot(q, start[k][1])), -1, 1)))
                drift.setdefault(o["name"], {})[marks[i]] = {"dp_mm": round(float(dp * 1000), 2),
                                                             "rot_deg": round(float(ang), 2)}
    speed = {o["name"]: float(np.linalg.norm(data.cvel[model.body(o["name"]).id][3:])) for o in objects}
    final = [pose(k) for k in range(B)]
    return {
        "hulls": hull_counts,
        "static_bodies": [o["name"] for k, o in enumerate(objects) if static[k]],
        "frame0_ncon": ncon0,
        "frame0_min_contact_dist_mm": {k: round(v * 1000, 2) for k, v in sorted(pen.items())},
        "drift_no_hands": drift,
        "final_speed_mps": speed,
        "final_pos": [f[0].tolist() for f in final],
        "final_wxyz": [f[1].tolist() for f in final],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path, nargs="?")
    ap.add_argument("--raw", type=Path, help="JSON {objects, pos0, quat0, boxes, static} (converter mode)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.01, help="physics step (sim/rl.yaml: 1/(20*5) = 0.01 s)")
    a = ap.parse_args()
    if a.raw:
        inp = json.loads(a.raw.read_text())
        rep = run_settle(inp["objects"], np.asarray(inp["pos0"]), np.asarray(inp["quat0"]),
                         [tuple(b) for b in inp["boxes"]], inp.get("static"), a.seconds, a.dt)
        (a.out or a.raw.with_suffix(".out.json")).write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: rep[k] for k in ("hulls", "static_bodies", "frame0_min_contact_dist_mm", "drift_no_hands")}))
        return 0
    import pyarrow.dataset as pads

    man = json.loads(a.manifest.read_text())
    t = pads.dataset(man["task_parquet"], format="parquet").to_table(columns=["object_body_position", "object_body_wxyz"])
    pos0 = np.asarray(t["object_body_position"][0].as_py()[0], float)
    quat0 = np.asarray(t["object_body_wxyz"][0].as_py()[0], float)
    rep = run_settle(man["objects"], pos0, quat0, support_boxes_from_usda(man["support"]["usda"]),
                     None, a.seconds, a.dt)
    show = {k: v for k, v in rep.items() if k not in ("final_pos", "final_wxyz", "frame0_ncon")}
    print(json.dumps(show, indent=1))
    a.manifest.with_name(a.manifest.stem + "_settle.json").write_text(json.dumps(show, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
