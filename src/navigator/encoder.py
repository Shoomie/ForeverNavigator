"""Lightweight visual encoder with color normalization, UI masking, and ONNX CPU runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


def build_default_ui_mask(height: int = 224, width: int = 224) -> np.ndarray:
    """Return a float32 mask (H, W) where 1.0 = keep 3D world, 0.0 = ignore typical WoW HUD."""
    mask = np.ones((height, width), dtype=np.float32)
    # Bottom action bars (bottom 15%)
    mask[int(height * 0.85) :, int(width * 0.12) : int(width * 0.88)] = 0.0
    # Top-right minimap corner
    mask[: int(height * 0.22), int(width * 0.80) :] = 0.0
    # Top-left unit frames
    mask[: int(height * 0.15), : int(width * 0.22)] = 0.0
    # Center-lower player character silhouette
    cy, cx = int(height * 0.60), width // 2
    cv2.ellipse(mask, (cx, cy), (int(width * 0.05), int(height * 0.12)), 0, 0, 360, 0.0, -1)
    return mask


def preprocess_frame(
    img_rgb: np.ndarray,
    size: int = 224,
    ui_mask: Optional[np.ndarray] = None,
    color_normalize: bool = True,
) -> np.ndarray:
    """Resize, color-normalize, apply UI mask, and format as NCHW float32 for ImageNet models."""
    if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
        img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    else:
        img = img_rgb.copy()

    if color_normalize:
        # Robust color/exposure normalization: blend 50% CLAHE-in-LAB with 50% original
        # so torchlight/gamma/monitor differences are dampened without losing material cues.
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        img = cv2.addWeighted(img, 0.45, img_eq, 0.55, 0)

    arr = img.astype(np.float32) / 255.0

    if ui_mask is not None:
        if ui_mask.shape != (size, size):
            m = cv2.resize(ui_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        else:
            m = ui_mask
        # Replace masked HUD pixels with neutral gray (ImageNet mean ~0.45)
        m3 = m[:, :, None]
        arr = arr * m3 + 0.45 * (1.0 - m3)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, :, :, :].astype(np.float32)


def export_backbone_to_onnx(
    onnx_path: str | Path,
    model_name: str = "mobilenetv3_small_100",
    input_size: int = 224,
) -> Path:
    """Export a lightweight timm vision backbone (feature extractor) to ONNX."""
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    class NormalizedEncoder(nn.Module):
        def __init__(self, backbone_name: str):
            super().__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.backbone(x)
            return F.normalize(feat, p=2, dim=-1)

    model = NormalizedEncoder(model_name).eval()
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return onnx_path


class ONNXVisualEncoder:
    """Ultra-lightweight CPU visual encoder backed by ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str | Path,
        input_size: int = 224,
        num_threads: int = 2,
    ):
        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
        self.default_mask = build_default_ui_mask(input_size, input_size)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def encode_single(
        self,
        img_rgb: np.ndarray,
        ui_mask: Optional[np.ndarray] = None,
        color_normalize: bool = True,
    ) -> np.ndarray:
        """Encode one RGB uint8 image into a 1D L2-normalized float32 embedding."""
        mask = self.default_mask if ui_mask is None else ui_mask
        inp = preprocess_frame(img_rgb, self.input_size, mask, color_normalize=color_normalize)
        emb = self.session.run([self.output_name], {self.input_name: inp})[0][0]
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def encode_batch(
        self,
        imgs_rgb: list[np.ndarray],
        ui_mask: Optional[np.ndarray] = None,
        color_normalize: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode a list of RGB uint8 images into (N, D) L2-normalized float32 embeddings."""
        mask = self.default_mask if ui_mask is None else ui_mask
        out_chunks = []
        for i in range(0, len(imgs_rgb), batch_size):
            batch_imgs = imgs_rgb[i : i + batch_size]
            batch_arr = np.concatenate(
                [
                    preprocess_frame(im, self.input_size, mask, color_normalize=color_normalize)
                    for im in batch_imgs
                ],
                axis=0,
            )
            embs = self.session.run([self.output_name], {self.input_name: batch_arr})[0]
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            out_chunks.append((embs / norms).astype(np.float32))
        return np.concatenate(out_chunks, axis=0)
