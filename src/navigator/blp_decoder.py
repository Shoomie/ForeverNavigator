"""Pure-Python BLP2 texture decoder (DXT1, DXT3, DXT5, Palettized, Uncompressed) -> PIL Image."""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np
from PIL import Image


def decode_blp(data: bytes) -> Image.Image:
    """Decode a WoW BLP2 texture into an RGBA PIL Image (mip level 0)."""
    if len(data) < 148 or data[:4] != b"BLP2":
        raise ValueError("Not a valid BLP2 file")

    (
        _type,
        color_encoding,
        alpha_depth,
        alpha_encoding,
        _has_mips,
        width,
        height,
    ) = struct.unpack_from("<IBBBBII", data, 4)

    mip_offsets = struct.unpack_from("<16I", data, 20)
    mip_sizes = struct.unpack_from("<16I", data, 84)
    offset = mip_offsets[0]
    size = mip_sizes[0]
    mip_data = data[offset : offset + size]

    if color_encoding == 1:
        # 256-color palettized (BGRA palette at offset 148, 1024 bytes)
        palette = np.frombuffer(data[148 : 148 + 1024], dtype=np.uint8).reshape(256, 4)
        # BGRA -> RGBA
        pal_rgba = palette[:, [2, 1, 0, 3]].copy()
        pal_rgba[:, 3] = 255
        n_pix = width * height
        indices = np.frombuffer(mip_data[:n_pix], dtype=np.uint8)
        rgba = pal_rgba[indices].reshape(height, width, 4)
        if alpha_depth == 8 and len(mip_data) >= n_pix * 2:
            rgba[:, :, 3] = np.frombuffer(mip_data[n_pix : n_pix * 2], dtype=np.uint8).reshape(
                height, width
            )
        elif alpha_depth == 1 and len(mip_data) >= n_pix + (n_pix + 7) // 8:
            bits = np.unpackbits(
                np.frombuffer(mip_data[n_pix : n_pix + (n_pix + 7) // 8], dtype=np.uint8),
                bitorder="little",
            )[:n_pix]
            rgba[:, :, 3] = (bits * 255).reshape(height, width)
        return Image.fromarray(rgba, mode="RGBA")

    if color_encoding == 2:
        # DXTC (BC1 / BC2 / BC3)
        if alpha_encoding == 0:
            rgba = _decode_dxt1(mip_data, width, height)
        elif alpha_encoding == 1:
            rgba = _decode_dxt3(mip_data, width, height)
        elif alpha_encoding == 7:
            rgba = _decode_dxt5(mip_data, width, height)
        else:
            rgba = _decode_dxt1(mip_data, width, height)
        return Image.fromarray(rgba, mode="RGBA")

    if color_encoding == 3:
        # Uncompressed BGRA
        arr = np.frombuffer(mip_data[: width * height * 4], dtype=np.uint8).reshape(
            height, width, 4
        )
        rgba = arr[:, :, [2, 1, 0, 3]].copy()
        return Image.fromarray(rgba, mode="RGBA")

    raise ValueError(f"Unsupported BLP2 color encoding: {color_encoding}")


def _rgb565_to_rgb888(c: np.ndarray) -> np.ndarray:
    r = ((c >> 11) & 0x1F) * 255 // 31
    g = ((c >> 5) & 0x3F) * 255 // 63
    b = (c & 0x1F) * 255 // 31
    return np.stack([r, g, b], axis=-1).astype(np.uint16)


def _decode_dxt_color_blocks(blocks: np.ndarray, has_1bit_alpha: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized DXT color block decoder. blocks shape: (N, 8) uint8."""
    c0 = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
    c1 = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
    rgb0 = _rgb565_to_rgb888(c0)  # (N, 3)
    rgb1 = _rgb565_to_rgb888(c1)  # (N, 3)

    gt = (c0 > c1) | (not has_1bit_alpha)
    gt3 = gt[:, None]

    rgb2 = np.where(gt3, (2 * rgb0 + rgb1) // 3, (rgb0 + rgb1) // 2)
    rgb3 = np.where(gt3, (rgb0 + 2 * rgb1) // 3, np.zeros_like(rgb0))

    palette = np.stack([rgb0, rgb1, rgb2, rgb3], axis=1).astype(np.uint8)  # (N, 4, 3)
    bits = (
        blocks[:, 4].astype(np.uint32)
        | (blocks[:, 5].astype(np.uint32) << 8)
        | (blocks[:, 6].astype(np.uint32) << 16)
        | (blocks[:, 7].astype(np.uint32) << 24)
    )
    shifts = np.arange(0, 32, 2, dtype=np.uint32)
    idx = ((bits[:, None] >> shifts[None, :]) & 3).astype(np.int32)  # (N, 16)

    n = blocks.shape[0]
    colors = palette[np.arange(n)[:, None], idx]  # (N, 16, 3)
    if has_1bit_alpha:
        alpha = np.where((~gt[:, None]) & (idx == 3), 0, 255).astype(np.uint8)
    else:
        alpha = np.full((n, 16), 255, dtype=np.uint8)
    return colors, alpha


def _reassemble_blocks(colors: np.ndarray, alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    rgba = np.concatenate([colors, alpha[:, :, None]], axis=-1)  # (bh*bw, 16, 4)
    rgba = rgba.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return rgba[:height, :width, :].copy()


def _decode_dxt1(data: bytes, width: int, height: int) -> np.ndarray:
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    n = bw * bh
    blocks = np.frombuffer(data[: n * 8], dtype=np.uint8).reshape(n, 8)
    colors, alpha = _decode_dxt_color_blocks(blocks, has_1bit_alpha=True)
    return _reassemble_blocks(colors, alpha, width, height)


def _decode_dxt3(data: bytes, width: int, height: int) -> np.ndarray:
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    n = bw * bh
    raw = np.frombuffer(data[: n * 16], dtype=np.uint8).reshape(n, 16)
    colors, _ = _decode_dxt_color_blocks(raw[:, 8:16], has_1bit_alpha=False)
    # 4-bit explicit alpha
    ab = raw[:, 0:8]
    lo = (ab & 0x0F) * 17
    hi = ((ab >> 4) & 0x0F) * 17
    alpha = np.empty((n, 16), dtype=np.uint8)
    alpha[:, 0::2] = lo
    alpha[:, 1::2] = hi
    return _reassemble_blocks(colors, alpha, width, height)


def _decode_dxt5(data: bytes, width: int, height: int) -> np.ndarray:
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    n = bw * bh
    raw = np.frombuffer(data[: n * 16], dtype=np.uint8).reshape(n, 16)
    colors, _ = _decode_dxt_color_blocks(raw[:, 8:16], has_1bit_alpha=False)

    a0 = raw[:, 0].astype(np.uint16)
    a1 = raw[:, 1].astype(np.uint16)
    gt = a0 > a1

    ap = np.empty((n, 8), dtype=np.uint8)
    ap[:, 0] = a0
    ap[:, 1] = a1
    ap[:, 2] = np.where(gt, (6 * a0 + 1 * a1) // 7, (4 * a0 + 1 * a1) // 5)
    ap[:, 3] = np.where(gt, (5 * a0 + 2 * a1) // 7, (3 * a0 + 2 * a1) // 5)
    ap[:, 4] = np.where(gt, (4 * a0 + 3 * a1) // 7, (2 * a0 + 3 * a1) // 5)
    ap[:, 5] = np.where(gt, (3 * a0 + 4 * a1) // 7, (1 * a0 + 4 * a1) // 5)
    ap[:, 6] = np.where(gt, (2 * a0 + 5 * a1) // 7, 0)
    ap[:, 7] = np.where(gt, (1 * a0 + 6 * a1) // 7, 255)

    bits = (
        raw[:, 2].astype(np.uint64)
        | (raw[:, 3].astype(np.uint64) << 8)
        | (raw[:, 4].astype(np.uint64) << 16)
        | (raw[:, 5].astype(np.uint64) << 24)
        | (raw[:, 6].astype(np.uint64) << 32)
        | (raw[:, 7].astype(np.uint64) << 40)
    )
    shifts = np.arange(0, 48, 3, dtype=np.uint64)
    a_idx = ((bits[:, None] >> shifts[None, :]) & 7).astype(np.int32)
    alpha = ap[np.arange(n)[:, None], a_idx]
    return _reassemble_blocks(colors, alpha, width, height)
