#!/usr/bin/env python3
"""Convert the official MANO / SMPL+H v1.2 pickles (Python-2 pickles holding chumpy objects) into chumpy-free pickles
with plain numpy / scipy.sparse values, same keys, so smplx / HaMeR / WiLoR load them without the (unmaintained)
chumpy package.  Loading uses a RESTRICTED unpickler: only numpy, scipy.sparse and chumpy array classes are allowed
(chumpy objects are replaced by their stored value).  The outputs stay under the MPI MANO licence (non-commercial
research): local use only, never commit or redistribute them.

  python -I mano_clean.py /mnt/secondary/v2d/weights/mano /mnt/secondary/v2d/weights/mano/clean
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


class _Chumpy:
    """Stand-in for chumpy.Ch & co: keep the pickled state, expose its array value."""
    def __setstate__(self, state):
        self.state = state

    def value(self):
        st = self.state if isinstance(self.state, dict) else {}
        if "idxs" in st and "a" in st:  # chumpy.reordering.Select: a.ravel()[idxs].reshape(preferred_shape)
            a = st["a"].value() if isinstance(st["a"], _Chumpy) else np.asarray(st["a"])
            out = np.asarray(a).ravel()[np.asarray(st["idxs"]).ravel()]
            return out.reshape(st["preferred_shape"]) if st.get("preferred_shape") else out
        for k in ("x", "_x", "a"):
            if k in st:
                v = st[k]
                return v.value() if isinstance(v, _Chumpy) else np.asarray(v)
        raise ValueError(f"chumpy object without array state: keys {sorted(st)[:12]}")


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Chumpy
        if module.split(".")[0] in ("numpy", "scipy") or (module, name) in (("copy_reg", "_reconstructor"),
                                                                             ("copyreg", "_reconstructor"),
                                                                             ("__builtin__", "object"),
                                                                             ("builtins", "object"),
                                                                             ("__builtin__", "set"),
                                                                             ("builtins", "set")):
            if module == "copy_reg":
                module = "copyreg"
            if module == "__builtin__":
                module = "builtins"
            if module.startswith("scipy.sparse.") and name.endswith("_matrix"):
                import scipy.sparse
                return getattr(scipy.sparse, name)
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def clean(src: Path, dst: Path) -> dict:
    with src.open("rb") as f:
        d = Restricted(f, encoding="latin1").load()
    out = {}
    for k, v in d.items():
        out[k] = v.value() if isinstance(v, _Chumpy) else v
    tmp = dst.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(out, f, protocol=4)
    tmp.replace(dst)
    return {k: (list(np.shape(v)) if hasattr(v, "shape") else type(v).__name__) for k, v in out.items()}


def main() -> int:
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl", "SMPLH_male.pkl", "SMPLH_female.pkl"):
        src = src_dir / name
        if not src.exists():
            continue
        shapes = clean(src, dst_dir / name)
        report[name] = dict(src_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                            out_sha256=hashlib.sha256((dst_dir / name).read_bytes()).hexdigest(), shapes=shapes)
    (dst_dir / "clean_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {kk: v["shapes"].get(kk) for kk in ("v_template", "shapedirs", "J_regressor", "weights",
                                                               "hands_components", "posedirs", "f")}
                      for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
