"""Minimal top-down RTMPose-m hand (OpenMMLab, Apache-2.0; hand5 = COCO-WholeBody-Hand + OneHand10K + FreiHAND2D +
RHD2D + Halpe-hand; 21 keypoints in the MediaPipe/OpenPose order) on onnxruntime CPU. No MANO.

  m = RTMPoseHand(); kps, scores = m(img, box_xyxy)     # img: HxW gray or HxWx3 BGR; kps (21,2) image px
Weights: /mnt/secondary/v2d/weights/rtmpose/rtmpose-m_hand5_256.onnx (sha256 39e85893...), from
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip
"""
import cv2
import numpy as np

ONNX = "/mnt/secondary/v2d/weights/rtmpose/rtmpose-m_hand5_256.onnx"
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class RTMPoseHand:
    def __init__(self, onnx=ONNX, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
        self.size = 256

    def __call__(self, img, box, padding=1.25):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x0, y0, x1, y1 = box
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        s = max(x1 - x0, y1 - y0) * padding
        S = self.size
        A = np.array([[S / s, 0, S / 2 - c[0] * S / s], [0, S / s, S / 2 - c[1] * S / s]], np.float32)
        crop = cv2.warpAffine(img, A, (S, S), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD
        sx, sy = self.sess.run(None, {"input": x.transpose(2, 0, 1)[None]})
        ix, iy = sx[0].argmax(-1), sy[0].argmax(-1)
        sc = np.minimum(sx[0].max(-1), sy[0].max(-1))
        u = np.stack([ix / 2.0, iy / 2.0], 1)  # simcc_split_ratio 2 -> input px
        kps = (u - [S / 2, S / 2]) * (s / S) + c
        return kps, sc
