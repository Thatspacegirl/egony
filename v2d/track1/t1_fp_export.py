#!/usr/bin/env python3
"""CARI4D stage 05 (prep/run_foundationpose_mhr_export.py, toolkit a709404e) with ONE degenerate case handled.

foundationpose_pose_selection.measure_visible_silhouettes leaves centroid_distance = +inf for a frame whose rendered
object is entirely hidden behind the human mask (or whose observed mask is empty).  The selection logic uses that inf
as "reject", which is fine, but the final validate_foundationpose_selection_arrays() refuses ANY inf in the saved
diagnostics and aborts the whole stage (val ep 5, a small vase held against the body: "generated FoundationPose
output/fp_centroid_distance contains infinity").  The toolkit's own convention for "no measurement" in these
diagnostics is NaN (run_foundationpose_mhr_export.py:262), so this wrapper converts +-inf -> NaN in the
fp_centroid_distance diagnostic right before that validation, i.e. after every pose decision was made.  Poses are
unchanged.  The toolkit clone is not modified (we only add files); arguments are passed through unchanged.
"""
import sys

import numpy as np

import prep.mhr_foundationpose_diagnostics as D  # noqa: E402  (PYTHONPATH = CARI4D SOURCE_ROOT, as for the stage)

_orig = D.validate_foundationpose_selection_arrays
_fixed = {"n": 0}


def _patched(data, expected_shape, label):
    if isinstance(data, dict) and "fp_centroid_distance" in data:
        v = np.asarray(data["fp_centroid_distance"])
        if np.issubdtype(v.dtype, np.floating) and np.isinf(v).any():
            _fixed["n"] += int(np.isinf(v).sum())
            data["fp_centroid_distance"] = np.where(np.isinf(v), np.nan, v).astype(v.dtype)
            print(f"[t1_fp_export] {label}: {int(np.isinf(v).sum())} inf centroid distances -> NaN", flush=True)
    return _orig(data, expected_shape, label)


D.validate_foundationpose_selection_arrays = _patched
import prep.run_foundationpose_mhr_export as M  # noqa: E402

M.validate_foundationpose_selection_arrays = _patched
try:
    import prep.mhr_packed_h5 as P  # noqa: E402
    P.validate_foundationpose_selection_arrays = _patched
except Exception:  # noqa: BLE001
    pass

if __name__ == "__main__":
    sys.argv[0] = M.__file__
    M.main()
