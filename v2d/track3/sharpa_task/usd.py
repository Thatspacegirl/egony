"""Support-surface USDA in the format flash_chord/scene/support.py reads (Cube/Cylinder, translate+scale)."""

from __future__ import annotations

from pathlib import Path


def write_support_usda(path: Path, center_xy, size_xy, top_z: float, thickness: float, name: str = "table") -> Path:
    """One axis-aligned Cube whose TOP face is at ``top_z`` (FlashCHORD ignores rotations: z must be up).

    flash_chord reads ``size`` (edge, default 2) * ``xformOp:scale`` as the full box dimensions and
    ``xformOp:translate`` as the box centre.
    """
    cx, cy = (float(v) for v in center_xy)
    sx, sy = (float(v) for v in size_xy)
    cz = float(top_z) - 0.5 * float(thickness)
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if safe[0].isdigit():
        safe = "_" + safe
    text = f"""#usda 1.0
(
    defaultPrim = "support_surfaces"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "support_surfaces"
{{
    def Cube "{safe}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        double size = 1
        color3f[] primvars:displayColor = [(0.55, 0.45, 0.35)]
        double3 xformOp:translate = ({cx:.9g}, {cy:.9g}, {cz:.9g})
        float3 xformOp:scale = ({sx:.9g}, {sy:.9g}, {float(thickness):.9g})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
}}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
