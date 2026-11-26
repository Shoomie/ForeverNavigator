"""Optional OpenGL 4.3 Compute Shader ZNCC Matcher for High-Precision Minimap Tracking."""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np

_COMPUTE_SRC = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D u_map;
layout(rgba32f, binding = 1) readonly uniform image2D u_tpl;
layout(r32f, binding = 2) writeonly uniform image2D u_out;

uniform ivec2 u_tpl_size;
uniform ivec2 u_out_size;
uniform vec4 u_tpl_mean;
uniform vec4 u_tpl_inv_norm;

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= u_out_size.x || gid.y >= u_out_size.y) {
        return;
    }

    int tw = u_tpl_size.x;
    int th = u_tpl_size.y;
    float n = float(tw * th);

    vec4 sum_m = vec4(0.0);
    vec4 sum_m2 = vec4(0.0);
    vec4 sum_mt = vec4(0.0);

    for (int j = 0; j < th; ++j) {
        for (int i = 0; i < tw; ++i) {
            vec4 mv = imageLoad(u_map, gid + ivec2(i, j));
            vec4 tv = imageLoad(u_tpl, ivec2(i, j));
            sum_m += mv;
            sum_m2 += mv * mv;
            sum_mt += mv * tv;
        }
    }

    vec4 mean_m = sum_m / n;
    vec4 var_m = max(sum_m2 - sum_m * mean_m, vec4(1e-7));
    vec4 num = sum_mt - sum_m * u_tpl_mean;
    vec4 ncc = num * inversesqrt(var_m) * u_tpl_inv_norm;

    // Channels: RGB = structural silhouette (0.70 weight), A = edge magnitude (0.30 weight)
    float sil_ncc = (ncc.r + ncc.g + ncc.b) / 3.0;
    float score = 0.70 * sil_ncc + 0.30 * ncc.a;
    imageStore(u_out, gid, vec4(score, 0.0, 0.0, 0.0));
}
"""


class GPUComputeMatcher:
    """OpenGL 4.3 compute-shader ZNCC template matcher (with automatic CPU fallback on error)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ctx = None
        self._prog = None
        self._map_textures: Dict[str, Tuple[object, int, int]] = {}
        self.renderer_name: str = ""
        self.available: bool = False
        try:
            import moderngl

            ctx = moderngl.create_standalone_context(require=430)
            self._prog = ctx.compute_shader(_COMPUTE_SRC)
            self._ctx = ctx
            self.renderer_name = str(ctx.info.get("GL_RENDERER", "OpenGL 4.3 GPU"))
            self.available = True
        except Exception:
            self._ctx = None
            self._prog = None
            self.available = False

    def upload_map(self, key: str, sil_rgb: np.ndarray, edge: np.ndarray) -> None:
        if not self.available or self._ctx is None:
            return
        h, w = sil_rgb.shape[:2]
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = sil_rgb
        rgba[:, :, 3] = edge
        with self._lock:
            old = self._map_textures.pop(key, None)
            if old is not None:
                try:
                    old[0].release()
                except Exception:
                    pass
            tex = self._ctx.texture((w, h), 4, data=np.ascontiguousarray(rgba).tobytes(), dtype="f4")
            self._map_textures[key] = (tex, w, h)

    def match(self, key: str, tpl_sil: np.ndarray, tpl_edge: np.ndarray) -> Optional[np.ndarray]:
        if not self.available or self._ctx is None or self._prog is None:
            return None
        with self._lock:
            entry = self._map_textures.get(key)
            if entry is None:
                return None
            map_tex, mw, mh = entry
            th, tw = tpl_sil.shape[:2]
            ow = mw - tw + 1
            oh = mh - th + 1
            if ow <= 0 or oh <= 0:
                return None

            t_rgba = np.empty((th, tw, 4), dtype=np.float32)
            t_rgba[:, :, :3] = tpl_sil
            t_rgba[:, :, 3] = tpl_edge
            flat = t_rgba.reshape(-1, 4)
            t_mean = flat.mean(axis=0)
            t_centered = flat - t_mean
            t_norm = np.sqrt(np.maximum(np.sum(t_centered * t_centered, axis=0), 1e-7))
            t_inv_norm = 1.0 / t_norm

            tpl_tex = self._ctx.texture((tw, th), 4, data=np.ascontiguousarray(t_rgba).tobytes(), dtype="f4")
            out_tex = self._ctx.texture((ow, oh), 1, dtype="f4")
            try:
                map_tex.bind_to_image(0, read=True, write=False)
                tpl_tex.bind_to_image(1, read=True, write=False)
                out_tex.bind_to_image(2, read=False, write=True)
                self._prog["u_tpl_size"].value = (tw, th)
                self._prog["u_out_size"].value = (ow, oh)
                self._prog["u_tpl_mean"].value = tuple(float(x) for x in t_mean)
                self._prog["u_tpl_inv_norm"].value = tuple(float(x) for x in t_inv_norm)
                gx = (ow + 15) // 16
                gy = (oh + 15) // 16
                self._prog.run(gx, gy, 1)
                self._ctx.finish()
                raw = out_tex.read()
                arr = np.frombuffer(raw, dtype=np.float32).reshape(oh, ow).copy()
                return np.nan_to_num(arr, nan=-1.0, posinf=-1.0, neginf=-1.0)
            finally:
                tpl_tex.release()
                out_tex.release()

    def close(self) -> None:
        with self._lock:
            for tex, _, _ in self._map_textures.values():
                try:
                    tex.release()
                except Exception:
                    pass
            self._map_textures.clear()
            if self._ctx is not None:
                try:
                    self._ctx.release()
                except Exception:
                    pass
                self._ctx = None
            self.available = False
