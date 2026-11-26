"""WoW WMO (World Map Object) binary parser for Root and Group files."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def iter_chunks(data: bytes, start: int = 0, end: Optional[int] = None):
    """Iterate over Blizzard reversed-FourCC chunks: (chunk_id_str, payload_bytes)."""
    if end is None:
        end = len(data)
    pos = start
    while pos + 8 <= end:
        magic = data[pos : pos + 4][::-1].decode("ascii", errors="replace")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        pos += 8
        if pos + size > end:
            break
        yield magic, data[pos : pos + size]
        pos += size


@dataclass
class WMOMaterial:
    flags: int
    shader: int
    blend_mode: int
    texture1_ref: int  # Either offset into MOTX or FileDataID (modern WoW)
    sidn_color: Tuple[int, int, int, int]
    diff_color: Tuple[int, int, int, int]


@dataclass
class WMOBatch:
    start_index: int
    count: int
    material_id: int


@dataclass
class WMOGroup:
    group_index: int
    flags: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    vertices: np.ndarray        # (V, 3) float32
    indices: np.ndarray         # (T*3,) uint32
    normals: np.ndarray         # (V, 3) float32
    uvs: np.ndarray             # (V, 2) float32
    vertex_colors: np.ndarray   # (V, 4) float32 RGBA in [0, 1]
    batches: List[WMOBatch] = field(default_factory=list)


@dataclass
class WMORoot:
    n_textures: int
    n_groups: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    materials: List[WMOMaterial]
    motx_raw: bytes
    group_fdids: List[int]  # From GFID chunk (modern WoW)
    uses_fdid_textures: bool


def parse_wmo_root(data: bytes) -> WMORoot:
    """Parse a WMO root file."""
    n_textures = 0
    n_groups = 0
    bbox_min = np.zeros(3, dtype=np.float32)
    bbox_max = np.zeros(3, dtype=np.float32)
    materials: List[WMOMaterial] = []
    motx_raw = b""
    group_fdids: List[int] = []

    for cid, payload in iter_chunks(data):
        if cid == "MOHD" and len(payload) >= 64:
            n_textures, n_groups = struct.unpack_from("<II", payload, 0)
            bmin = struct.unpack_from("<3f", payload, 36)
            bmax = struct.unpack_from("<3f", payload, 48)
            bbox_min = np.array(bmin, dtype=np.float32)
            bbox_max = np.array(bmax, dtype=np.float32)
        elif cid == "MOTX":
            motx_raw = payload
        elif cid == "MOMT":
            # Each SMOMaterial is 64 bytes
            count = len(payload) // 64
            for i in range(count):
                off = i * 64
                flags, shader, blend_mode, tex1 = struct.unpack_from("<IIII", payload, off)
                sidn = struct.unpack_from("<4B", payload, off + 16)
                diff = struct.unpack_from("<4B", payload, off + 28)
                materials.append(
                    WMOMaterial(
                        flags=flags,
                        shader=shader,
                        blend_mode=blend_mode,
                        texture1_ref=tex1,
                        sidn_color=sidn,
                        diff_color=diff,
                    )
                )
        elif cid == "GFID":
            # Array of uint32 FileDataIDs for groups (n_groups * LODs; first n_groups are LOD 0)
            cnt = len(payload) // 4
            all_ids = list(struct.unpack_from(f"<{cnt}I", payload, 0))
            group_fdids = all_ids[:n_groups] if n_groups > 0 else all_ids

    # If MOTX is empty or tiny and materials have large texture1_ref, they are FileDataIDs
    uses_fdid_textures = len(motx_raw) == 0 and any(m.texture1_ref > 0 for m in materials)
    return WMORoot(
        n_textures=n_textures,
        n_groups=n_groups,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        materials=materials,
        motx_raw=motx_raw,
        group_fdids=group_fdids,
        uses_fdid_textures=uses_fdid_textures,
    )


def resolve_material_texture(
    root: WMORoot,
    mat: WMOMaterial,
    path_to_id: Dict[str, int],
) -> Optional[int]:
    """Return the FileDataID for a WMO material's primary diffuse texture, if available."""
    ref = mat.texture1_ref
    if ref == 0:
        return None
    if root.uses_fdid_textures or ref >= len(root.motx_raw):
        return ref
    end = root.motx_raw.find(b"\x00", ref)
    if end == -1:
        end = len(root.motx_raw)
    tex_path = root.motx_raw[ref:end].decode("ascii", errors="ignore").replace("\\", "/").lower()
    if not tex_path:
        return None
    return path_to_id.get(tex_path)


def parse_wmo_group(data: bytes, group_index: int = 0) -> Optional[WMOGroup]:
    """Parse a WMO group file (contains MOGP header + sub-chunks)."""
    mogp_payload = None
    for cid, payload in iter_chunks(data):
        if cid == "MOGP":
            mogp_payload = payload
            break
    if mogp_payload is None or len(mogp_payload) < 68:
        return None

    flags = struct.unpack_from("<I", mogp_payload, 8)[0]
    bmin = np.array(struct.unpack_from("<3f", mogp_payload, 12), dtype=np.float32)
    bmax = np.array(struct.unpack_from("<3f", mogp_payload, 24), dtype=np.float32)

    vertices = np.zeros((0, 3), dtype=np.float32)
    indices = np.zeros((0,), dtype=np.uint32)
    normals = np.zeros((0, 3), dtype=np.float32)
    uvs = np.zeros((0, 2), dtype=np.float32)
    vertex_colors: Optional[np.ndarray] = None
    batches: List[WMOBatch] = []
    mopy_flags: Optional[np.ndarray] = None

    for cid, payload in iter_chunks(mogp_payload, start=68):
        if cid == "MOVT":
            vertices = np.frombuffer(payload, dtype="<f4").reshape(-1, 3).copy()
        elif cid == "MOVI":
            indices = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
        elif cid == "MONR":
            normals = np.frombuffer(payload, dtype="<f4").reshape(-1, 3).copy()
        elif cid == "MOTV" and len(uvs) == 0:
            uvs = np.frombuffer(payload, dtype="<f4").reshape(-1, 2).copy()
        elif cid == "MOCV" and vertex_colors is None:
            # BGRA uint8 -> RGBA float32
            bgra = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 4)
            rgba = bgra[:, [2, 1, 0, 3]].astype(np.float32) / 255.0
            vertex_colors = rgba
        elif cid == "MOPY":
            # 2 bytes per triangle: (flags, material_id)
            mopy_flags = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 2)
        elif cid == "MOBA":
            # Each SMOBatch is 24 bytes:
            # 12B bbox (6 int16), 4B start_index (u32), 2B count (u16),
            # 2B min_index, 2B max_index, 1B flags, 1B material_id
            nb = len(payload) // 24
            for i in range(nb):
                off = i * 24
                start_idx, count = struct.unpack_from("<IH", payload, off + 12)
                mat_id = payload[off + 23]
                if count > 0:
                    batches.append(WMOBatch(start_index=start_idx, count=count, material_id=mat_id))

    if len(vertices) == 0 or len(indices) == 0:
        return None

    n_verts = len(vertices)
    if len(normals) != n_verts:
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_verts, 1))
    if len(uvs) != n_verts:
        uvs = np.zeros((n_verts, 2), dtype=np.float32)
    if vertex_colors is None or len(vertex_colors) != n_verts:
        vertex_colors = np.full((n_verts, 4), 0.75, dtype=np.float32)
        vertex_colors[:, 3] = 1.0

    # If no MOBA batches, create a single batch for all triangles
    if not batches and len(indices) >= 3:
        mat_id = int(mopy_flags[0, 1]) if mopy_flags is not None and len(mopy_flags) > 0 else 0
        batches.append(WMOBatch(start_index=0, count=len(indices), material_id=mat_id))

    return WMOGroup(
        group_index=group_index,
        flags=flags,
        bbox_min=bmin,
        bbox_max=bmax,
        vertices=vertices,
        indices=indices,
        normals=normals,
        uvs=uvs,
        vertex_colors=vertex_colors,
        batches=batches,
    )
