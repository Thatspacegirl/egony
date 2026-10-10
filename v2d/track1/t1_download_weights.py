#!/usr/bin/env python3
"""Download the (newly accessible, 2026-10-09) gated models Track 1 uses, into /mnt/secondary/v2d/weights/.

Layouts follow the toolkit's own download_weights.py so the toolkit code finds them unchanged:
  cari4d      : /mnt/secondary/v2d/weights/cari4d/  == v2d_cari4d/lib/download_weights.py layout
                  cari4d/2026-08-25-09-35-57/{step200000.pth,resolved_config.yaml,manifest.json}  (nvidia/cari4d_commercial @ 1f7287ac)
                  sam3d_body/checkpoints/sam-3d-body-dinov3/{model.ckpt,assets/mhr_model.pt,...} (facebook/sam-3d-body-dinov3 @ 11aaa346)
                  hf_home/hub/models--Ruicheng--moge-2-vitl-normal (@ b135031b)
                  sam3d_body/torch_home/hub/facebookresearch_dinov3_main (@ 6876159a), facebookresearch_dinov2_main (@ 7764ea0f)
                  sam3d_body/torch_home/hub/checkpoints/dinov2_vit{b,s}14_pretrain.pth (sha256-checked)
                  foundationpose -> symlink to /mnt/secondary/v2d/weights/foundationpose (nvlabs_pytorch, already present)
  sam3d_objects: /mnt/secondary/v2d/weights/sam3d_objects/ == v2d_sam3d/lib/download_weights.py layout
                  hf-download/ (facebook/sam-3d-objects), hf_home/ (Ruicheng/moge-vitl), torch_home/hub/checkpoints (DINOv2 reg4)
  trellis2    : HF cache ($HF_HOME): microsoft/TRELLIS.2-4B, facebook/dinov3-vitl16-pretrain-lvd1689m, briaai/RMBG-2.0

  python t1_download_weights.py cari4d sam3d_objects trellis2
Licences are recorded in ../track1/README.md ("Models and licences").
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

W = Path("/mnt/secondary/v2d/weights")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def git_checkout(dest: Path, url: str, rev: str):
    if (dest / ".git").is_dir():
        cur = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if cur == rev:
            return
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth=1", "origin", rev], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", rev], check=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", rev], check=True)


def fetch(url, dest: Path, want_sha=None):
    if dest.is_file() and (want_sha is None or sha256(dest) == want_sha):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if want_sha and sha256(tmp) != want_sha:
        raise SystemExit(f"sha mismatch {url}")
    os.replace(tmp, dest)


def cari4d():
    from huggingface_hub import snapshot_download
    root = W / "cari4d"
    run = "2026-08-25-09-35-57"
    snapshot_download("nvidia/cari4d_commercial", revision="1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9",
                      allow_patterns=[f"{run}/step200000.pth", f"{run}/resolved_config.yaml", f"{run}/manifest.json"],
                      local_dir=root / "cari4d")
    ck = root / "cari4d" / run / "step200000.pth"
    s = sha256(ck)
    assert s == "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3", s
    print("cari4d ok", s, flush=True)
    snapshot_download("facebook/sam-3d-body-dinov3", revision="11aaa346c7204874a1cbafe3d39a979080b2c55a",
                      local_dir=root / "sam3d_body" / "checkpoints" / "sam-3d-body-dinov3")
    print("sam-3d-body-dinov3 ok", flush=True)
    snapshot_download("Ruicheng/moge-2-vitl-normal", revision="b135031bae30b5ac2ae141a0e68717795ce38340",
                      cache_dir=root / "hf_home" / "hub")
    print("moge2 ok", flush=True)
    th = root / "sam3d_body" / "torch_home" / "hub"
    git_checkout(th / "facebookresearch_dinov3_main", "https://github.com/facebookresearch/dinov3.git",
                 "6876159a11b4df116f30f667f8c9888617df0751")
    git_checkout(th / "facebookresearch_dinov2_main", "https://github.com/facebookresearch/dinov2.git",
                 "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    for name, sh in {"dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
                     "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"}.items():
        arch = name.split("_")[1]
        fetch(f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{arch}/{name}", th / "checkpoints" / name, sh)
    fp = root / "foundationpose"
    if not fp.exists():
        fp.symlink_to(W / "foundationpose")
    print("cari4d weights complete", flush=True)


def sam3d_objects():
    from huggingface_hub import snapshot_download
    root = W / "sam3d_objects"
    snapshot_download("facebook/sam-3d-objects", local_dir=root / "hf-download")
    print("sam-3d-objects ok", flush=True)
    snapshot_download("Ruicheng/moge-vitl", cache_dir=root / "hf_home" / "hub")
    print("moge-vitl ok", flush=True)
    for name in ("dinov2_vitl14_reg4_pretrain.pth", "dinov2_vitb14_reg4_pretrain.pth"):
        arch = name.split("_")[1]
        fetch(f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{arch}/{name}", root / "torch_home" / "hub" / "checkpoints" / name)
    print("sam3d_objects weights complete", flush=True)


def trellis2():
    from huggingface_hub import snapshot_download
    for r, pats in (("microsoft/TRELLIS.2-4B", None),
                    ("facebook/dinov3-vitl16-pretrain-lvd1689m", None),
                    ("briaai/RMBG-2.0", ["*.py", "*.json", "model.safetensors", "README.md", "LICENSE*"])):
        p = snapshot_download(r, allow_patterns=pats)
        print(r, p, flush=True)
    print("trellis2 weights complete", flush=True)


if __name__ == "__main__":
    for what in sys.argv[1:]:
        {"cari4d": cari4d, "sam3d_objects": sam3d_objects, "trellis2": trellis2}[what]()
