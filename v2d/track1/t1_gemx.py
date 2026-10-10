#!/usr/bin/env python3
"""Track 1 human stage: GEM-X (nvidia/GEM-X, ungated) on one video with our K, plus the per-frame MHR output of
GEM-X's own SAM-3D-Body stage re-decoded with our K.

Env: /mnt/secondary/v2d/envs/gemx (source v2d_env.sh). GPU, under the shared lock.

Everything GEM-X does is ../track2/tier3/gemx_run.py (unchanged numbers).  This wrapper only hooks GEM-X's
SAM3DBExtractor.extract_video_features: GEM-X runs SAM-3D-Body (the `sam3d_body.ckpt` shipped in nvidia/GEM-X,
Meta SAM License) with ITS default intrinsics (f = image diagonal = 1920 px) to make GEM's input tokens, and
throws away the MHR head output.  We keep that output and additionally re-run ONLY the decoder (the backbone
image embedding does not depend on the intrinsics; the camera enters through the ray conditioning and the CLIFF
condition of the decoder) with our K, which gives per-frame native MHR parameters in our camera.

Extra output  <out>/<stem>/sam3db_mhr.npz  (T = video frames):
  K (3,3); bbx_xys (T,3)
  k_mhr_model_params (T,204)  k_shape (T,45)  k_cam_t (T,3) m  k_kp2d (T,70,2) px  k_kp3d (T,70,3) m (no transl)
  k_global_rot (T,3)  k_hand (T,108)  k_scale (T,28)
  d_mhr_model_params, d_shape, d_cam_t   (same with GEM-X's default intrinsics)
MHR convention (sam_3d_body mhr_head): verts_cam = MHR(shape, model_params)[cm]/100 * diag(1,-1,-1) + cam_t,
with model_params[:3] (root translation) = 0.

Usage: python t1_gemx.py --video V --out OUT --K-values fx,fy,cx,cy [--bbx bbx.pt]   (any gemx_run.py option)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T3 = HERE.parent / "track2" / "tier3"
sys.path.insert(0, str(T3))
import gemx_run  # noqa: E402

GEMX, SOMA013 = gemx_run.GEMX, gemx_run.SOMA013


def _arg(name):
    v = sys.argv
    return v[v.index(name) + 1] if name in v else None


def main():
    sys.path.insert(0, str(GEMX))
    sys.path.insert(0, str(SOMA013))
    import soma as _soma  # noqa: F401  (must resolve to SOMA-X 0.1.3 before GEM-X imports it)
    assert Path(_soma.__file__).resolve().is_relative_to(SOMA013), _soma.__file__
    import torch
    import gem.utils.sam3db_extractor as s3m

    kv = _arg("--K-values")
    assert kv, "t1_gemx.py needs --K-values fx,fy,cx,cy"
    fx, fy, cx, cy = [float(x) for x in kv.split(",")]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    video = Path(_arg("--video")).resolve()
    od = Path(_arg("--out")).resolve() / video.stem
    rec = {k: [] for k in ("k_mhr_model_params", "k_shape", "k_cam_t", "k_kp2d", "k_kp3d", "k_global_rot", "k_hand",
                           "k_scale", "d_mhr_model_params", "d_shape", "d_cam_t")}

    def extract_video_features(self, video_path, bbx_xys, img_ds=1.0, batch_size=16, render_mhr=False):
        # == GEM-X's SAM3DBExtractor.extract_video_features (same calls, same order) + a K-conditioned decoder pass
        from sam_3d_body.data.utils.prepare_batch import prepare_batch
        from sam_3d_body.utils import recursive_to
        assert img_ds == 1.0
        imgs = s3m.read_video_np(video_path, scale=img_ds)
        bbx = torch.as_tensor(bbx_xys).float().cpu().numpy()
        model = self.estimator.model
        tokens, transls = [], []
        for i in range(0, len(imgs), batch_size):
            mb = []
            for j, img in enumerate(imgs[i:i + batch_size]):
                c_x, c_y, s = bbx[i + j]
                box = np.array([c_x - s / 2, c_y - s / 2, c_x + s / 2, c_y + s / 2], np.float32).reshape(1, 4)
                mb.append(prepare_batch(img, self.estimator.transform, box, masks=None, masks_score=None))
            cb = {}
            for key in mb[0]:
                v0 = mb[0][key]
                if isinstance(v0, torch.Tensor):
                    cb[key] = torch.cat([d[key] for d in mb])
                elif isinstance(v0, np.ndarray):
                    cb[key] = np.concatenate([d[key] for d in mb])
                else:
                    cb[key] = [d[key] for d in mb]
            cb = recursive_to(cb, self.device)
            model._initialize_batch(cb)
            with torch.no_grad():
                out = model.forward_step(cb, decoder_type="body")
                pt = out.get("pose_token", None)
                assert pt is not None, "pose_token not reconstructed"
                pt = pt.detach().float().cpu()
                if pt.ndim == 3:
                    pt = pt[:, 0]
                m = out["mhr"]
                tokens.append(pt)
                transls.append(m["pred_cam_t"].detach().float().cpu())
                rec["d_mhr_model_params"].append(m["mhr_model_params"].float().cpu())
                rec["d_shape"].append(m["shape"].float().cpu())
                rec["d_cam_t"].append(m["pred_cam_t"].float().cpu())
                # -- decoder only, our K (backbone embedding reused)
                b2 = dict(cb)
                b2["cam_int"] = K[None].to(cb["img"]).expand(cb["img"].shape[0], 3, 3).contiguous()
                ray = model._flatten_person(model.get_ray_condition(b2))
                bt = model.cfg.MODEL.BACKBONE.TYPE
                if bt in ["vit_hmr", "vit", "vit_b", "vit_l"]:
                    ray = ray[:, :, :, 32:-32]
                elif bt in ["vit_hmr_512_384"]:
                    ray = ray[:, :, :, 64:-64]
                idx = model.body_batch_idx
                b2["ray_cond"] = ray[idx].clone()
                cond = model._get_decoder_condition(b2)
                B, N = cb["img"].shape[:2]
                kp = torch.zeros((B * N, 1, 3)).to(cb["img"]); kp[:, :, -1] = -2
                _, po = model.forward_decoder(out["image_embeddings"][idx], init_estimate=None, keypoints=kp[idx],
                                              prev_estimate=None, condition_info=cond[idx], batch=b2)
                po = po[-1]
                rec["k_mhr_model_params"].append(po["mhr_model_params"].float().cpu())
                rec["k_shape"].append(po["shape"].float().cpu())
                rec["k_cam_t"].append(po["pred_cam_t"].float().cpu())
                rec["k_kp2d"].append(po["pred_keypoints_2d"].float().cpu())
                rec["k_kp3d"].append(po["pred_keypoints_3d"].float().cpu())
                rec["k_global_rot"].append(po["global_rot"].float().cpu())
                rec["k_hand"].append(po["hand"].float().cpu())
                rec["k_scale"].append(po["scale"].float().cpu())
        arrs = {k: torch.cat(v).numpy().astype(np.float32) for k, v in rec.items()}
        assert all(len(v) == len(imgs) for v in arrs.values()), {k: v.shape for k, v in arrs.items()}
        assert all(np.isfinite(v).all() for v in arrs.values())
        (od).mkdir(parents=True, exist_ok=True)
        tmp = od / "sam3db_mhr.tmp.npz"
        np.savez(tmp, K=K.numpy(), bbx_xys=bbx.astype(np.float32), **arrs)
        os.replace(tmp, od / "sam3db_mhr.npz")
        print("wrote", od / "sam3db_mhr.npz", flush=True)
        return {"pose_tokens": torch.cat(tokens, 0), "transls": torch.cat(transls, 0), "rendered_imgs": []}

    s3m.SAM3DBExtractor.extract_video_features = extract_video_features
    gemx_run.main()


if __name__ == "__main__":
    main()
