"""ForeverNavigator GPU-Accelerated PySide6 + ModernGL Desktop UI.

Renders the 3-level prestitched 2D map pyramid (4K Azeroth -> 4K Continent L1 -> Native CASC Tiles
and active Zone maps) directly as hardware-mipmapped OpenGL textures with crisp QPainter vector
overlays, achieving 120+ FPS pan & smooth cursor-anchored stepless zoom with <1% CPU overhead.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QSurfaceFormat,
    QWheelEvent,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..combatlog.parser import iter_combatlog_file
from ..combatlog.stats import analyze_combat_events
from ..navigator.dev_logger import DevGameplayLogger
from ..navigator.fast_capture import FastScreenCapturer
from ..navigator.gpu_viewer3d import GPU3DViewer, find_wow_dir
from ..navigator.minimap_tracker import MinimapROI, MinimapTracker, load_roi, save_roi
from ..navigator.settings import (
    AppSettings,
    detect_storage_type,
    load_settings,
    resolve_storage_mode,
    save_settings,
)
from ..navigator.spawns import get_spawns_for_dungeon
from ..navigator.tracker import LightweightDungeonStore, TrackerEstimate
from ..net.client import ForeverNavigatorClient
from ..net.protocol import (
    WOW_CLASS_COLORS,
    PlayerIdentity,
    PlayerState,
    SyncPreferences,
    get_local_lan_ip,
    now_ms,
)
from ..net.server import ForeverNavigatorServer, HostConfig
from ..replay.session_store import SessionStore
from ..replay.timeline import TimelineController
from ..tactics.tools import RulerMeasurement, measure_yards, yards_to_map_px
from ..tactics.whiteboard import TacticalWhiteboard, WhiteboardOp

DARK_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #0d1117;
    color: #f0f6fc;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 9pt;
}
QFrame#Surface {
    background-color: #161b22;
    border: 1px solid #30363d;
}
QFrame#Card {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
}
QLabel#CardTitle {
    color: #8b949e;
    font-size: 7.5pt;
    font-weight: bold;
}
QLabel#Pill {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
    font-size: 8pt;
}
QPushButton {
    background-color: #21262d;
    color: #f0f6fc;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: bold;
}
QPushButton:hover { background-color: #30363d; }
QPushButton:checked { background-color: #1f6feb; border-color: #58a6ff; color: white; }
QPushButton#Primary { background-color: #1f6feb; border-color: #388bfd; color: white; }
QPushButton#Primary:hover { background-color: #388bfd; }
QPushButton#Success { background-color: #238636; border-color: #2ea043; color: white; }
QPushButton#Success:hover { background-color: #2ea043; }
QPushButton#Danger { background-color: #da3633; border-color: #f85149; color: white; }
QPushButton#Purple { background-color: #8957e5; border-color: #a371f7; color: white; }
QLineEdit, QComboBox, QListWidget, QTreeWidget, QTextEdit {
    background-color: #161b22;
    color: #f0f6fc;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 3px 6px;
}
QListWidget::item:selected, QTreeWidget::item:selected {
    background-color: #1f6feb;
    color: white;
}
QTabWidget::pane {
    border: none;
    background-color: #161b22;
}
QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    padding: 6px 10px;
    font-weight: bold;
    border: none;
}
QTabBar::tab:selected {
    background-color: #21262d;
    color: #58a6ff;
}
QHeaderView::section {
    background-color: #21262d;
    color: #8b949e;
    border: none;
    padding: 3px 4px;
    font-weight: bold;
    font-size: 8pt;
}
QCheckBox { spacing: 5px; }
"""

_QUAD_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
uniform vec4 u_rect;     // (x0, y0, x1, y1) in NDC [-1..1]
uniform vec4 u_uv_rect;  // (u0, v0, u1, v1)
out vec2 v_uv;
void main() {
    vec2 p = mix(u_rect.xy, u_rect.zw, in_pos);
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv = mix(u_uv_rect.xy, u_uv_rect.zw, in_uv);
}
"""

_QUAD_FRAG = """
#version 330 core
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""

_CIRCLE_VERT = """
#version 330 core
in vec2 in_pos;
uniform vec4 u_rect;     // (x0, y0, x1, y1) in NDC [-1..1]
out vec2 v_local;        // [-1..1] across quad
void main() {
    vec2 p = mix(u_rect.xy, u_rect.zw, in_pos);
    gl_Position = vec4(p, 0.0, 1.0);
    v_local = in_pos * 2.0 - 1.0;
}
"""

_CIRCLE_FRAG = """
#version 330 core
in vec2 v_local;
uniform vec4 u_fill;
uniform vec4 u_stroke;
uniform float u_outer_px;
uniform float u_stroke_px;
uniform int u_dashed;
out vec4 fragColor;
void main() {
    float dist_px = length(v_local) * u_outer_px;
    float aa = 1.0;
    float outer_alpha = 1.0 - smoothstep(u_outer_px - aa, u_outer_px + aa * 0.5, dist_px);
    if (outer_alpha <= 0.001) {
        discard;
    }
    float inner_r = max(0.0, u_outer_px - u_stroke_px);
    float stroke_mask = smoothstep(inner_r - aa, inner_r + aa * 0.5, dist_px);
    if (u_dashed == 1) {
        float ang = atan(v_local.y, v_local.x);
        float dash_count = max(12.0, floor(u_outer_px * 0.45));
        float phase = fract((ang / 6.2831853 + 0.5) * dash_count);
        float dash_mask = smoothstep(0.0, 0.08, phase) * (1.0 - smoothstep(0.55, 0.63, phase));
        stroke_mask *= dash_mask;
    }
    vec4 col = mix(u_fill, u_stroke, stroke_mask);
    col.a *= outer_alpha;
    if (col.a <= 0.001) {
        discard;
    }
    fragColor = col;
}
"""

_LINE_VERT = """
#version 330 core
in vec2 in_pos;          // NDC xy
in vec4 in_col;          // RGBA
in vec2 in_edge;         // (signed_perp_px, half_width_px)
in vec2 in_dash;         // (along_px, dash_period_px; 0 = solid)
out vec4 v_col;
out vec2 v_edge;
out vec2 v_dash;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_col = in_col;
    v_edge = in_edge;
    v_dash = in_dash;
}
"""

_LINE_FRAG = """
#version 330 core
in vec4 v_col;
in vec2 v_edge;
in vec2 v_dash;
out vec4 fragColor;
void main() {
    float d = abs(v_edge.x);
    float hw = v_edge.y;
    float alpha = 1.0 - smoothstep(max(0.0, hw - 0.75), hw + 0.75, d);
    if (v_dash.y > 0.5) {
        float phase = fract(v_dash.x / v_dash.y);
        alpha *= step(phase, 0.58);
    }
    if (alpha * v_col.a <= 0.001) {
        discard;
    }
    fragColor = vec4(v_col.rgb, v_col.a * alpha);
}
"""


def _role_badge(role: str) -> str:
    r = (role or "DPS").upper()
    return {"LEADER": "L", "OFFICER": "O", "TANK": "T", "HEALER": "H"}.get(r, "D")


def _to_rgba_u8(arr: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA/Gray uint8 image to contiguous HxWx4 RGBA for zero-repack GPU texture upload."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA), dtype=np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA), dtype=np.uint8)


_COLOR_CACHE: Dict[str, Tuple[float, float, float, float]] = {}


def _hex_to_rgba_f(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    key = hex_color if alpha == 1.0 else f"{hex_color}:{alpha:.2f}"
    cached = _COLOR_CACHE.get(key)
    if cached is not None:
        return cached
    qc = QColor(hex_color)
    if not qc.isValid():
        res = (1.0, 0.35, 0.27, alpha)
    else:
        res = (qc.redF(), qc.greenF(), qc.blueF(), alpha * qc.alphaF())
    _COLOR_CACHE[key] = res
    return res


class _GPUOverlayRenderer:
    """Pure ModernGL 2D overlay renderer (circles, lines, polylines, LRU-cached crisp text quads).

    Renders all 2D/3D vector overlays directly in ModernGL without touching QOpenGLWidget with
    QPainter (preserving Qt+ModernGL isolation) and without a translucent child QWidget (which
    otherwise costs 8-25 ms/frame in full-viewport CPU backing-store uploads on Windows).
    """

    def __init__(self, ctx: Any, quad_prog: Any, quad_vao: Any, quad_vbo: Any):
        self.ctx = ctx
        self._quad_prog = quad_prog
        self._quad_vao = quad_vao
        self._circle_prog = ctx.program(vertex_shader=_CIRCLE_VERT, fragment_shader=_CIRCLE_FRAG)
        self._circle_vao = ctx.vertex_array(self._circle_prog, [(quad_vbo, "2f 2x4", "in_pos")])
        self._line_prog = ctx.program(vertex_shader=_LINE_VERT, fragment_shader=_LINE_FRAG)
        self._line_vbo_cap = 4096 * 10 * 4  # initial 160 KB dynamic buffer
        self._line_vbo = ctx.buffer(reserve=self._line_vbo_cap, dynamic=True)
        self._line_vao = ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 4f 2f 2f", "in_pos", "in_col", "in_edge", "in_dash")],
        )
        self._line_verts: List[float] = []
        self._labels: OrderedDict[Tuple[str, int, bool, str, bool, int], Tuple[Any, float, float]] = OrderedDict()
        self._label_cap = 256

    def release(self) -> None:
        for tex, _, _ in self._labels.values():
            try:
                tex.release()
            except Exception:
                pass
        self._labels.clear()
        for obj in (self._circle_vao, self._circle_prog, self._line_vao, self._line_vbo, self._line_prog):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass

    def begin_lines(self) -> None:
        self._line_verts.clear()

    def add_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        cw: float,
        ch: float,
        rgba: Tuple[float, float, float, float],
        width_px: float = 2.0,
        dash_px: float = 0.0,
        along_offset: float = 0.0,
    ) -> float:
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-4:
            return along_offset
        hw = max(0.5, width_px * 0.5)
        pad = hw + 1.0
        nx = -dy / seg_len
        ny = dx / seg_len
        px = nx * pad
        py = ny * pad
        inv_w = 2.0 / max(1.0, cw)
        inv_h = 2.0 / max(1.0, ch)

        ax0 = (x0 + px) * inv_w - 1.0
        ay0 = 1.0 - (y0 + py) * inv_h
        bx0 = (x0 - px) * inv_w - 1.0
        by0 = 1.0 - (y0 - py) * inv_h
        ax1 = (x1 + px) * inv_w - 1.0
        ay1 = 1.0 - (y1 + py) * inv_h
        bx1 = (x1 - px) * inv_w - 1.0
        by1 = 1.0 - (y1 - py) * inv_h

        r, g, b, a = rgba
        d0 = along_offset
        d1 = along_offset + seg_len
        self._line_verts.extend((
            ax0, ay0, r, g, b, a,  pad, hw, d0, dash_px,
            bx0, by0, r, g, b, a, -pad, hw, d0, dash_px,
            ax1, ay1, r, g, b, a,  pad, hw, d1, dash_px,
            bx0, by0, r, g, b, a, -pad, hw, d0, dash_px,
            bx1, by1, r, g, b, a, -pad, hw, d1, dash_px,
            ax1, ay1, r, g, b, a,  pad, hw, d1, dash_px,
        ))
        return d1

    def add_polyline(
        self,
        pts: List[Tuple[float, float]],
        cw: float,
        ch: float,
        rgba: Tuple[float, float, float, float],
        width_px: float = 2.0,
        dash_px: float = 0.0,
    ) -> None:
        along = 0.0
        for i in range(len(pts) - 1):
            along = self.add_line(
                pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                cw, ch, rgba, width_px=width_px, dash_px=dash_px, along_offset=along,
            )

    def flush_lines(self) -> None:
        if not self._line_verts:
            return
        import moderngl

        arr = np.asarray(self._line_verts, dtype=np.float32)
        n_bytes = arr.nbytes
        if n_bytes > self._line_vbo_cap:
            self._line_vbo_cap = max(n_bytes * 2, self._line_vbo_cap * 2)
            self._line_vbo.orphan(size=self._line_vbo_cap)
        self._line_vbo.write(arr)
        self._line_vao.render(moderngl.TRIANGLES, vertices=len(self._line_verts) // 10)
        self._line_verts.clear()

    def draw_circle(
        self,
        cx: float,
        cy: float,
        radius_px: float,
        cw: float,
        ch: float,
        fill_rgba: Tuple[float, float, float, float],
        stroke_rgba: Tuple[float, float, float, float],
        stroke_px: float = 1.5,
        dashed: bool = False,
    ) -> None:
        import moderngl

        outer = max(1.0, radius_px + stroke_px * 0.5)
        pad = outer + 1.5
        if cx + pad < 0.0 or cx - pad > cw or cy + pad < 0.0 or cy - pad > ch:
            return
        inv_w = 2.0 / max(1.0, cw)
        inv_h = 2.0 / max(1.0, ch)
        nx0 = (cx - pad) * inv_w - 1.0
        nx1 = (cx + pad) * inv_w - 1.0
        ny0 = 1.0 - (cy + pad) * inv_h
        ny1 = 1.0 - (cy - pad) * inv_h
        self._circle_prog["u_rect"].value = (nx0, ny0, nx1, ny1)
        self._circle_prog["u_fill"].value = fill_rgba
        self._circle_prog["u_stroke"].value = stroke_rgba
        self._circle_prog["u_outer_px"].value = float(pad)
        self._circle_prog["u_stroke_px"].value = float(stroke_px + (pad - outer))
        self._circle_prog["u_dashed"].value = 1 if dashed else 0
        self._circle_vao.render(moderngl.TRIANGLE_STRIP)

    def _get_label_tex(
        self,
        text: str,
        pt_size: int,
        bold: bool,
        hex_color: str,
        shadow: bool,
        dpr: float,
    ) -> Tuple[Any, float, float]:
        import moderngl

        dpr_q = max(100, int(round(dpr * 100)))
        key = (text, pt_size, bold, hex_color, shadow, dpr_q)
        cached = self._labels.get(key)
        if cached is not None:
            self._labels.move_to_end(key)
            return cached

        font = QFont("Segoe UI", pt_size, QFont.Weight.Bold if bold else QFont.Weight.Normal)
        fm = QFontMetrics(font)
        br = fm.boundingRect(text)
        pad_x = 4
        pad_y = 2
        lw = max(2, br.width() + pad_x * 2 + 2)
        lh = max(2, fm.height() + pad_y * 2)
        scale = dpr_q / 100.0
        pw = max(2, int(math.ceil(lw * scale)))
        ph = max(2, int(math.ceil(lh * scale)))

        img = QImage(pw, ph, QImage.Format.Format_RGBA8888)
        img.fill(Qt.GlobalColor.transparent)
        img.setDevicePixelRatio(scale)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.setFont(font)
        baseline_y = pad_y + fm.ascent()
        if shadow:
            p.setPen(QPen(QColor(10, 12, 16, 220)))
            p.drawText(QPointF(pad_x + 1.0, baseline_y + 1.0), text)
        p.setPen(QPen(QColor(hex_color)))
        p.drawText(QPointF(float(pad_x), float(baseline_y)), text)
        p.end()

        raw_bits = bytes(img.constBits())
        bpl = int(img.bytesPerLine())
        if bpl == pw * 4:
            tex_data = raw_bits
        else:
            arr_img = np.frombuffer(raw_bits, dtype=np.uint8).reshape((ph, bpl))[:, : pw * 4]
            tex_data = np.ascontiguousarray(arr_img)
        tex = self.ctx.texture((pw, ph), 4, data=tex_data)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._labels[key] = (tex, float(lw), float(lh))
        while len(self._labels) > self._label_cap:
            _, (old_tex, _, _) = self._labels.popitem(last=False)
            try:
                old_tex.release()
            except Exception:
                pass
        return tex, float(lw), float(lh)

    def draw_text(
        self,
        x_px: float,
        y_px: float,
        text: str,
        cw: float,
        ch: float,
        dpr: float,
        hex_color: str = "#f0f6fc",
        pt_size: int = 8,
        bold: bool = True,
        shadow: bool = True,
        align_center: bool = False,
    ) -> None:
        import moderngl

        if not text:
            return
        tex, lw, lh = self._get_label_tex(text, pt_size, bold, hex_color, shadow, dpr)
        if align_center:
            x0 = x_px - lw * 0.5
            y0 = y_px - lh * 0.5
        else:
            x0 = x_px - 4.0
            y0 = y_px - lh * 0.72
        x1 = x0 + lw
        y1 = y0 + lh
        if x1 < 0.0 or x0 > cw or y1 < 0.0 or y0 > ch:
            return
        inv_w = 2.0 / max(1.0, cw)
        inv_h = 2.0 / max(1.0, ch)
        nx0 = x0 * inv_w - 1.0
        nx1 = x1 * inv_w - 1.0
        ny0 = 1.0 - y1 * inv_h
        ny1 = 1.0 - y0 * inv_h
        tex.use(location=0)
        self._quad_prog["u_tex"].value = 0
        self._quad_prog["u_rect"].value = (nx0, ny0, nx1, ny1)
        self._quad_prog["u_uv_rect"].value = (0.0, 0.0, 1.0, 1.0)
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)


class GPUMapViewport(QOpenGLWidget):
    """Hardware-accelerated OpenGL 2D/3D map viewport with GPU mipmaps + GPU-native vector overlays."""

    def __init__(self, win: "ForeverNavigatorQtWindow"):
        super().__init__(win)
        self.win = win
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._mgl = None
        self._prog = None
        self._vao = None
        self._vbo = None
        self._gpu_ov: Optional[_GPUOverlayRenderer] = None
        self._textures: OrderedDict[str, Tuple[Any, int, int, int, int]] = OrderedDict()  # key -> (tex, w, h, ver, comp)
        self._tex_versions: Dict[str, int] = {}
        self._pending_uploads: Dict[str, Tuple[np.ndarray, bool]] = {}  # key -> (rgba_arr, build_mips)
        self._pending_evictions: List[str] = []
        self._pending_3d_scene: Optional[Tuple[str, Dict[str, Any]]] = None
        self._cached_3d_spawns: Dict[str, Tuple[np.ndarray, np.ndarray, List[Any]]] = {}

    def queue_texture_upload(self, key: str, arr_rgb: np.ndarray, build_mips: bool = True) -> None:
        """Schedule an RGB/RGBA uint8 numpy array to be uploaded to GPU VRAM on the next paintGL."""
        self._pending_uploads[key] = (_to_rgba_u8(arr_rgb), build_mips)
        self._tex_versions[key] = self._tex_versions.get(key, 0) + 1
        self.update()

    def queue_texture_upload_rgba(self, key: str, arr_rgba: np.ndarray, build_mips: bool = True) -> None:
        """Schedule a pre-converted contiguous RGBA uint8 array for zero-copy GPU upload."""
        self._pending_uploads[key] = (arr_rgba, build_mips)
        self._tex_versions[key] = self._tex_versions.get(key, 0) + 1
        self.update()

    def queue_texture_evict(self, key: str) -> None:
        self._pending_uploads.pop(key, None)
        self._pending_evictions.append(key)

    def has_texture(self, key: str) -> bool:
        return key in self._textures or key in self._pending_uploads

    def release_gl_resources(self) -> None:
        if self._mgl is None:
            return
        try:
            self.makeCurrent()
            self.win.viewer_3d.release_gl_only()
            if self._gpu_ov is not None:
                self._gpu_ov.release()
                self._gpu_ov = None
            for tex, _, _, _, _ in self._textures.values():
                try:
                    tex.release()
                except Exception:
                    pass
            self._textures.clear()
            for obj in (self._vao, self._vbo, self._prog):
                if obj is not None:
                    try:
                        obj.release()
                    except Exception:
                        pass
            self._vao = None
            self._vbo = None
            self._prog = None
            self.doneCurrent()
        except Exception:
            pass

    def initializeGL(self) -> None:
        import moderngl

        self._mgl = moderngl.create_context()
        self._prog = self._mgl.program(vertex_shader=_QUAD_VERT, fragment_shader=_QUAD_FRAG)
        verts = np.array(
            [
                0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 0.0,
            ],
            dtype=np.float32,
        )
        self._vbo = self._mgl.buffer(verts)
        self._vao = self._mgl.vertex_array(self._prog, [(self._vbo, "2f 2f", "in_pos", "in_uv")])
        self._gpu_ov = _GPUOverlayRenderer(self._mgl, self._prog, self._vao, self._vbo)

    def _flush_texture_ops(self) -> None:
        import moderngl

        if self._mgl is None:
            return
        while self._pending_evictions:
            k = self._pending_evictions.pop()
            old = self._textures.pop(k, None)
            if old is not None:
                try:
                    old[0].release()
                except Exception:
                    pass

        for key, (arr, build_mips) in list(self._pending_uploads.items()):
            self._pending_uploads.pop(key, None)
            h, w = arr.shape[:2]
            comp = 4 if (arr.ndim == 3 and arr.shape[2] == 4) else 3
            old = self._textures.pop(key, None)
            if old is not None and old[1] == w and old[2] == h and old[4] == comp:
                tex = old[0]
                tex.write(arr)
            else:
                if old is not None:
                    try:
                        old[0].release()
                    except Exception:
                        pass
                tex = self._mgl.texture((w, h), comp, data=arr)
                tex.repeat_x = False
                tex.repeat_y = False
            if build_mips:
                tex.build_mipmaps()
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                try:
                    tex.anisotropy = 8.0
                except Exception:
                    pass
            else:
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._textures[key] = (tex, w, h, self._tex_versions.get(key, 1), comp)
            self._textures.move_to_end(key)

        # Enforce GPU texture budget (SSD=6, HDD=10 non-pinned textures to avoid continent/zone thrashing)
        limit = 10 if self.win._effective_storage_mode == "hdd" else 6
        pinned = {"__azeroth_ov", "__deep_tile", "__view_3d"}
        evictable = [k for k in self._textures.keys() if k not in pinned]
        while len(evictable) > limit:
            ek = evictable.pop(0)
            old = self._textures.pop(ek, None)
            if ek.startswith("zone_"):
                self.win._fullres_uploaded_zones.discard(ek[5:])
            if old is not None:
                try:
                    old[0].release()
                except Exception:
                    pass

    def _draw_textured_quad(
        self,
        tex: Any,
        x0_px: float,
        y0_px: float,
        x1_px: float,
        y1_px: float,
        cw: float,
        ch: float,
        uv_rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ) -> None:
        import moderngl

        if self._prog is None or self._vao is None:
            return
        nx0 = (x0_px / max(1.0, cw)) * 2.0 - 1.0
        nx1 = (x1_px / max(1.0, cw)) * 2.0 - 1.0
        ny0 = 1.0 - (y1_px / max(1.0, ch)) * 2.0
        ny1 = 1.0 - (y0_px / max(1.0, ch)) * 2.0
        tex.use(location=0)
        self._prog["u_tex"].value = 0
        self._prog["u_rect"].value = (nx0, ny0, nx1, ny1)
        self._prog["u_uv_rect"].value = uv_rect
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def paintGL(self) -> None:
        cw = float(max(1, self.width()))
        ch = float(max(1, self.height()))
        dpr = float(self.devicePixelRatioF())
        fb_w = max(1, int(round(cw * dpr)))
        fb_h = max(1, int(round(ch * dpr)))

        if self._mgl is not None:
            self._flush_texture_ops()
            fbo = self._mgl.detect_framebuffer()
            fbo.use()
            fbo.viewport = (0, 0, fb_w, fb_h)
            fbo.scissor = None
            self._mgl.disable(self._mgl.DEPTH_TEST | self._mgl.CULL_FACE | self._mgl.BLEND)
            if self.win._showing_azeroth_overview and not self.win.view_3d_mode and not self.win.replay_mode:
                fbo.clear(18 / 255.0, 36 / 255.0, 58 / 255.0, 1.0)
                self._paint_overview_gl(cw, ch)
            elif self.win.view_3d_mode:
                self._paint_3d_gl(fbo, cw, ch)
            else:
                fbo.clear(13 / 255.0, 17 / 255.0, 23 / 255.0, 1.0)
                self._paint_zone_gl(cw, ch)

            # GPU-native vector & text overlays (zero QPainter on GL context, zero CPU backing-store upload)
            if self._gpu_ov is not None:
                self._mgl.disable(self._mgl.DEPTH_TEST | self._mgl.CULL_FACE)
                self._mgl.enable(self._mgl.BLEND)
                self._mgl.blend_func = (self._mgl.SRC_ALPHA, self._mgl.ONE_MINUS_SRC_ALPHA)
                self._paint_overlays_gpu(self._gpu_ov, cw, ch, dpr)

    def _paint_overlays_gpu(self, ov: _GPUOverlayRenderer, cw: float, ch: float, dpr: float) -> None:
        if self.win._showing_azeroth_overview and not self.win.view_3d_mode and not self.win.replay_mode:
            if "__azeroth_ov" not in self._textures:
                ov.draw_text(
                    cw * 0.5, ch * 0.5,
                    "Loading Azeroth Overview (Kalimdor & Eastern Kingdoms)...",
                    cw, ch, dpr, hex_color="#8b949e", pt_size=11, bold=True, shadow=False, align_center=True,
                )
        elif self.win.view_3d_mode:
            self._paint_3d_overlays_gpu(ov, cw, ch, dpr)
        else:
            self._paint_zone_overlays_gpu(ov, cw, ch, dpr)

    def _paint_overview_gl(self, cw: float, ch: float) -> None:
        ov = self.win._azeroth_overview
        if ov is None:
            return
        sc_ov = self.win._overview_eff_scale(int(cw), int(ch))

        # Check if zoomed in enough for Level-1 4K continent or Level-2 async native CASC tiles
        drew_hi = False
        if sc_ov > 1.15 and self.win.world_maps is not None:
            mapped = ov.overview_px_to_continent_px(self.win._overview_pan_x, self.win._overview_pan_y)
            if mapped is not None:
                _folder, cont_slug, cx_cont, cy_cont, cont_per_ov = mapped
                sc_cont = sc_ov / max(1e-6, cont_per_ov)

                # Level 2: Async native 512px CASC viewport patch when deeply zoomed & settled
                if not self.win._is_interacting and sc_cont >= 1.65:
                    tile_key = (cont_slug, int(round(cx_cont)), int(round(cy_cont)), round(sc_cont, 3), int(cw), int(ch))
                    if self.win._deep_tile_key == tile_key and "__deep_tile" in self._textures:
                        tex = self._textures["__deep_tile"][0]
                        self._draw_textured_quad(tex, 0.0, 0.0, cw, ch, cw, ch)
                        drew_hi = True
                    else:
                        self.win._request_deep_tiles_async(cont_slug, cx_cont, cy_cont, sc_cont, int(cw), int(ch))

                # Level 1: Prestitched 4096x4096 WoW: Forever continent GPU texture (0.1 ms GPU draw!)
                if not drew_hi:
                    l1_key = f"__l1_{cont_slug}"
                    self.win._ensure_continent_l1_async(cont_slug)
                    if l1_key in self._textures and cont_slug in self.win._cont_l1_native:
                        tex = self._textures[l1_key][0]
                        native_px = float(self.win._cont_l1_native[cont_slug])
                        x0 = (0.0 - cx_cont) * sc_cont + cw * 0.5
                        y0 = (0.0 - cy_cont) * sc_cont + ch * 0.5
                        x1 = (native_px - cx_cont) * sc_cont + cw * 0.5
                        y1 = (native_px - cy_cont) * sc_cont + ch * 0.5
                        self._draw_textured_quad(tex, x0, y0, x1, y1, cw, ch)
                        drew_hi = True

        # Level 0: 4K Azeroth world GPU texture
        if not drew_hi and "__azeroth_ov" in self._textures:
            tex = self._textures["__azeroth_ov"][0]
            x0 = (0.0 - self.win._overview_pan_x) * sc_ov + cw * 0.5
            y0 = (0.0 - self.win._overview_pan_y) * sc_ov + ch * 0.5
            x1 = (float(ov.width) - self.win._overview_pan_x) * sc_ov + cw * 0.5
            y1 = (float(ov.height) - self.win._overview_pan_y) * sc_ov + ch * 0.5
            self._draw_textured_quad(tex, x0, y0, x1, y1, cw, ch)

    def _paint_3d_gl(self, fbo: Any, cw: float, ch: float) -> None:
        if self._mgl is None:
            return
        if self._pending_3d_scene is not None:
            slug_p, prepared = self._pending_3d_scene
            self._pending_3d_scene = None
            self.win.viewer_3d.upload_prepared_scene(self._mgl, slug_p, prepared)

        slug = self.win._active_slug()
        d = self.win.tracker_3d.dungeons.get(slug)
        if d is None or self.win.viewer_3d._loaded_slug != slug or self.win.viewer_3d._scene is None:
            fbo.clear(18 / 255.0, 21 / 255.0, 28 / 255.0, 1.0)
            self.win._request_3d_scene_async(slug)
            return

        self.win.viewer_3d.render_direct_gl(
            self._mgl,
            fbo,
            slug,
            d,
            int(max(1, cw)),
            int(max(1, ch)),
            player_world=self.win._latest_est.world_pos if self.win._latest_est else None,
            follow_player=self.win.chk_follow.isChecked(),
        )

    def _paint_3d_overlays_gpu(self, ov: _GPUOverlayRenderer, cw: float, ch: float, dpr: float) -> None:
        slug = self.win._active_slug()
        d = self.win.tracker_3d.dungeons.get(slug)
        v3d = self.win.viewer_3d
        err = getattr(self.win, "_scene_3d_error", None)
        if err:
            ov.draw_text(cw * 0.5, ch * 0.5, err, cw, ch, dpr, hex_color="#e6c373", pt_size=11, bold=True, align_center=True)
            return
        if d is None or v3d._loaded_slug != slug or v3d._scene is None or v3d._last_mvp is None:
            name = d.name if d is not None else slug
            ov.draw_text(
                cw * 0.5, ch * 0.5, f"Loading 3D WMO Model ({name})...",
                cw, ch, dpr, hex_color="#8b949e", pt_size=11, bold=True, align_center=True,
            )
            return

        mvp_mat = v3d._last_mvp

        def project(pts: np.ndarray):
            ones = np.ones((len(pts), 1), dtype=np.float32)
            clip = np.hstack([pts, ones]) @ mvp_mat
            w_c = clip[:, 3]
            valid = w_c > 0.5
            ndc_x = np.where(valid, clip[:, 0] / np.maximum(0.5, w_c), -99.0)
            ndc_y = np.where(valid, clip[:, 1] / np.maximum(0.5, w_c), -99.0)
            return (ndc_x * 0.5 + 0.5) * cw, (0.5 - ndc_y * 0.5) * ch, valid

        ov.begin_lines()
        circles_to_draw: List[Tuple[float, float, float, Tuple[float, float, float, float], Tuple[float, float, float, float], float]] = []
        labels_to_draw: List[Tuple[float, float, str, str]] = []

        # Cached 3D spawn positions (avoids per-frame nearest-waypoint search!)
        if self.win.chk_spawns.isChecked():
            cached_sp = self._cached_3d_spawns.get(slug)
            if cached_sp is None:
                spawns = get_spawns_for_dungeon(slug)
                if spawns:
                    map_h = d.map_size_px
                    sp_pts = []
                    for sp in spawns:
                        wx, wy = d.map_px_to_world(sp.u * map_h, sp.v * map_h)
                        wz = v3d.snap_world_z(d, np.array([wx, wy], dtype=np.float32))
                        sp_pts.append([wx, wy, wz])
                    sp_arr = np.array(sp_pts, dtype=np.float32)
                    sp_top = sp_arr.copy()
                    sp_top[:, 2] += 14.0
                    cached_sp = (sp_arr, sp_top, spawns)
                else:
                    cached_sp = (np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), [])
                self._cached_3d_spawns[slug] = cached_sp

            sp_arr, sp_top, spawns = cached_sp
            if spawns:
                bu, bv, bok = project(sp_arr)
                pu, pv, pok = project(sp_top)
                for i, sp in enumerate(spawns):
                    if not (bok[i] and pok[i]):
                        continue
                    bx, by, px, py = float(bu[i]), float(bv[i]), float(pu[i]), float(pv[i])
                    if not (-40.0 <= px <= cw + 40.0 and -40.0 <= py <= ch + 40.0):
                        continue
                    hex_c = "#5adcfc" if sp.kind == "entrance" else "#ffa53c"
                    rgba_c = _hex_to_rgba_f(hex_c)
                    ov.add_line(bx, by, px, py, cw, ch, rgba_c, width_px=2.0)
                    circles_to_draw.append((px, py, 5.0, rgba_c, (1.0, 1.0, 1.0, 1.0), 1.2))
                    labels_to_draw.append((px + 8.0, py + 4.0, sp.label, hex_c))

        # Local player 3D beacon
        if v3d._last_p3d is not None:
            p3d = v3d._last_p3d
            bu, bv, bok = project(p3d[None, :])
            tu, tv, tok = project((p3d + np.array([0.0, 0.0, 18.0], dtype=np.float32))[None, :])
            if bok[0] and tok[0]:
                bx, by, tx, ty = float(bu[0]), float(bv[0]), float(tu[0]), float(tv[0])
                ov.add_line(bx, by, tx, ty, cw, ch, _hex_to_rgba_f("#ff644b"), width_px=3.0)
                circles_to_draw.append((tx, ty, 7.0, _hex_to_rgba_f("#ff5041"), (1.0, 1.0, 1.0, 1.0), 2.0))

        ov.flush_lines()
        for cx, cy, rad, fill_c, stroke_c, sw in circles_to_draw:
            ov.draw_circle(cx, cy, rad, cw, ch, fill_c, stroke_c, stroke_px=sw)
        for lx, ly, txt, hex_c in labels_to_draw:
            ov.draw_text(lx, ly, txt, cw, ch, dpr, hex_color=hex_c, pt_size=8, bold=True)

        ov.draw_text(
            10.0, ch - 8.0,
            f"GPU 3D Model (WMO) | Left-drag: Orbit | Right-drag: Pan | Scroll: Zoom ({int(v3d.zoom * 100)}%)",
            cw, ch, dpr, hex_color="#c8d2e1", pt_size=8, bold=False,
        )

    def _paint_zone_gl(self, cw: float, ch: float) -> None:
        slug = self.win._active_slug()
        d = self.win.tracker_3d.dungeons.get(slug)
        if d is None:
            return
        map_h = float(d.map_size_px)
        sc = self.win._eff_scale(int(cw), int(ch), int(map_h))
        dx0 = (0.0 - self.win.pan_x) * sc + cw * 0.5
        dy0 = (0.0 - self.win.pan_y) * sc + ch * 0.5
        dx1 = (map_h - self.win.pan_x) * sc + cw * 0.5
        dy1 = (map_h - self.win.pan_y) * sc + ch * 0.5
        zone_covers = (dx0 <= 0 and dy0 <= 0 and dx1 >= cw and dy1 >= ch)

        # Optional lightweight continent complement behind zone when zoomed out
        if not zone_covers and self.win.world_maps is not None:
            entry = self.win.world_maps.get_entry(slug)
            folder = entry.minimap_folder if entry is not None else None
            if folder and entry is not None and entry.category != "Continents" and d.map_rot90 == 1:
                cont_slug = "kalimdor" if folder == "Kalimdor" else ("eastern_kingdoms" if folder == "Azeroth" else None)
                if cont_slug:
                    self.win._ensure_continent_l1_async(cont_slug)
                    l1_key = f"__l1_{cont_slug}"
                    cont_meta = self.win._cont_meta.get(cont_slug)
                    if l1_key in self._textures and cont_meta is not None:
                        tex = self._textures[l1_key][0]
                        wx_c, wy_c = d.map_px_to_world(self.win.pan_x, self.win.pan_y)
                        ch_cont = float(cont_meta["map_size_px"])
                        c_scale = float(cont_meta["map_scale"])
                        ox_c, oy_c = cont_meta["map_origin_xy"]
                        cx_cont = (ch_cont - 1.0) - (wy_c - oy_c) * c_scale
                        cy_cont = (ch_cont - 1.0) - (wx_c - ox_c) * c_scale
                        sc_cont = sc * (d.map_scale / max(1e-6, c_scale))
                        cx0 = (0.0 - cx_cont) * sc_cont + cw * 0.5
                        cy0 = (0.0 - cy_cont) * sc_cont + ch * 0.5
                        cx1 = (ch_cont - cx_cont) * sc_cont + cw * 0.5
                        cy1 = (ch_cont - cy_cont) * sc_cont + ch * 0.5
                        self._draw_textured_quad(tex, cx0, cy0, cx1, cy1, cw, ch)

        # Active zone texture (streams full-res from SSD in background while showing preview immediately)
        self.win._ensure_zone_texture(slug, d)
        z_key = f"zone_{slug}"
        if z_key in self._textures:
            tex = self._textures[z_key][0]
            self._draw_textured_quad(tex, dx0, dy0, dx1, dy1, cw, ch)

    def _paint_zone_overlays_gpu(self, ov: _GPUOverlayRenderer, cw: float, ch: float, dpr: float) -> None:
        slug = self.win._active_slug()
        d = self.win.tracker_3d.dungeons.get(slug)
        if d is None:
            return
        map_h = float(d.map_size_px)
        sc = self.win._eff_scale(int(cw), int(ch), int(map_h))
        pan_x = self.win.pan_x
        pan_y = self.win.pan_y
        half_cw = cw * 0.5
        half_ch = ch * 0.5

        ov.begin_lines()
        labels: List[Tuple[float, float, str, str, int]] = []
        range_circles: List[Tuple[float, float, float, Tuple[float, float, float, float]]] = []

        # Collaborative Whiteboard ops
        for op in self.win.whiteboard.ops_for(slug):
            rgba_c = _hex_to_rgba_f(op.color)
            if op.kind == "stroke" and len(op.points) >= 2:
                pts = [((u * map_h - pan_x) * sc + half_cw, (v * map_h - pan_y) * sc + half_ch) for u, v in op.points]
                ov.add_polyline(pts, cw, ch, rgba_c, width_px=float(max(1, op.width)))
            elif op.kind == "text" and op.points:
                sx = (op.points[0][0] * map_h - pan_x) * sc + half_cw
                sy = (op.points[0][1] * map_h - pan_y) * sc + half_ch
                labels.append((sx, sy, op.text, op.color, 10))
            elif op.kind == "circle_range" and op.points:
                sx = (op.points[0][0] * map_h - pan_x) * sc + half_cw
                sy = (op.points[0][1] * map_h - pan_y) * sc + half_ch
                r_px = yards_to_map_px(op.radius_yd, d.map_scale) * sc
                range_circles.append((sx, sy, r_px, rgba_c))
                labels.append((sx - 18.0, sy - r_px - 6.0, f"{op.radius_yd:.0f} yd", op.color, 8))

        # Active in-progress stroke
        if len(self.win._active_stroke_pts) >= 2:
            pts = [((u * map_h - pan_x) * sc + half_cw, (v * map_h - pan_y) * sc + half_ch) for u, v in self.win._active_stroke_pts]
            ov.add_polyline(pts, cw, ch, _hex_to_rgba_f("#ffe066"), width_px=3.0)

        # Active Yard Ruler
        if self.win.active_ruler is not None:
            r = self.win.active_ruler
            x0 = (r.start_uv[0] * map_h - pan_x) * sc + half_cw
            y0 = (r.start_uv[1] * map_h - pan_y) * sc + half_ch
            x1 = (r.end_uv[0] * map_h - pan_x) * sc + half_cw
            y1 = (r.end_uv[1] * map_h - pan_y) * sc + half_ch
            ov.add_line(x0, y0, x1, y1, cw, ch, _hex_to_rgba_f("#63b3ed"), width_px=2.0, dash_px=12.0)
            labels.append(((x0 + x1) * 0.5, (y0 + y1) * 0.5 - 10.0, f"{r.yards:.1f} yd", "#90cdf4", 10))

        # Dead-player X lines queued into same single line batch
        shown = self.win.replay_players if self.win.replay_mode else self.win.raid_players
        dead_rgba = _hex_to_rgba_f("#e53e3e")
        for pl in shown.values():
            if pl.dungeon_slug != slug or bool(getattr(pl, "alive", True)):
                continue
            sx = (pl.map_px[0] - pan_x) * sc + half_cw
            sy = (pl.map_px[1] - pan_y) * sc + half_ch
            ov.add_line(sx - 5.0, sy - 5.0, sx + 5.0, sy + 5.0, cw, ch, dead_rgba, width_px=2.0)
            ov.add_line(sx - 5.0, sy + 5.0, sx + 5.0, sy - 5.0, cw, ch, dead_rgba, width_px=2.0)

        ov.flush_lines()

        # Dashed range circles
        transparent = (0.0, 0.0, 0.0, 0.0)
        for sx, sy, r_px, rgba_c in range_circles:
            ov.draw_circle(sx, sy, r_px, cw, ch, transparent, rgba_c, stroke_px=2.0, dashed=True)

        # Spawns
        if self.win.chk_spawns.isChecked():
            dark_outline = _hex_to_rgba_f("#14161c")
            for sp in get_spawns_for_dungeon(slug):
                sx = (sp.u * map_h - pan_x) * sc + half_cw
                sy = (sp.v * map_h - pan_y) * sc + half_ch
                if not (-40.0 <= sx <= cw + 40.0 and -40.0 <= sy <= ch + 40.0):
                    continue
                hex_c = "#5adcfc" if sp.kind == "entrance" else "#ffa53c"
                ov.draw_circle(sx, sy, 5.0, cw, ch, _hex_to_rgba_f(hex_c), dark_outline, stroke_px=1.5)
                labels.append((sx + 8.0, sy + 4.0, sp.label, hex_c, 8))

        # Connected Raid Players / Replay Players
        for pl in shown.values():
            if pl.dungeon_slug != slug:
                continue
            sx = (pl.map_px[0] - pan_x) * sc + half_cw
            sy = (pl.map_px[1] - pan_y) * sc + half_ch
            alive = bool(getattr(pl, "alive", True))
            hex_c = WOW_CLASS_COLORS.get(pl.wow_class, "#ff5a46") if alive else "#878c96"
            is_local = (pl.player_id == self.win.identity.player_id) and not self.win.replay_mode
            outline = _hex_to_rgba_f("#f5c96b" if is_local else "#ffffff")
            rad = 7.0 if is_local else 6.0
            ov.draw_circle(sx, sy, rad, cw, ch, _hex_to_rgba_f(hex_c), outline, stroke_px=2.0)
            badge = _role_badge(pl.role)
            label = f"[{badge}|DEAD] {pl.name}" if not alive else f"[{badge}] {pl.name}"
            labels.append((sx + 9.0, sy + 3.0, label, "#fc8181" if not alive else hex_c, 8))

        # Local player marker (clean dot using sub-pixel smoothed coordinates)
        est = self.win._latest_est
        if est is not None and not self.win.replay_mode and self.win.identity.player_id not in shown:
            rpx = getattr(self.win, "_render_px", None)
            if rpx is not None:
                sx = (rpx[0] - pan_x) * sc + half_cw
                sy = (rpx[1] - pan_y) * sc + half_ch
            else:
                sx = (est.map_px[0] - pan_x) * sc + half_cw
                sy = (est.map_px[1] - pan_y) * sc + half_ch
            ov.draw_circle(sx, sy, 7.0, cw, ch, _hex_to_rgba_f("#ff5a46"), (1.0, 1.0, 1.0, 1.0), stroke_px=2.0)

        # Crisp cached text labels
        for lx, ly, txt, hex_c, pt in labels:
            ov.draw_text(lx, ly, txt, cw, ch, dpr, hex_color=hex_c, pt_size=pt, bold=True)

    # --- Mouse & Wheel Forwarding ---
    def wheelEvent(self, event: QWheelEvent) -> None:
        self.win._on_mousewheel(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.win._on_mouse_press(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.win._on_mouse_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.win._on_mouse_release(event)


class QtROIPickerOverlay(QWidget):
    """Translucent fullscreen overlay over the chosen monitor to drag a box around the WoW minimap."""

    def __init__(self, mon: dict, on_done, screen=None):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.on_done = on_done
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowOpacity(0.32)
        self.setStyleSheet("background-color: #0a0c10;")
        self.setCursor(Qt.CursorShape.CrossCursor)
        if screen is not None:
            geom = screen.geometry()
            self.setGeometry(geom.x(), geom.y(), geom.width(), geom.height())
        else:
            dpr = float(self.devicePixelRatioF()) or 1.0
            self.setGeometry(
                int(round(int(mon["left"]) / dpr)),
                int(round(int(mon["top"]) / dpr)),
                int(round(int(mon["width"]) / dpr)),
                int(round(int(mon["height"]) / dpr)),
            )
        self._start: Optional[QPointF] = None
        self._cur: Optional[QPointF] = None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position()
            self._cur = event.position()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start is not None:
            self._cur = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._start is not None and self._cur is not None:
            x0, x1 = sorted([self._start.x(), self._cur.x()])
            y0, y1 = sorted([self._start.y(), self._cur.y()])
            lw = float(max(1, self.width()))
            lh = float(max(1, self.height()))
            if x1 - x0 >= 20 and y1 - y0 >= 20:
                self.on_done(
                    MinimapROI(
                        float(np.clip(x0 / lw, 0.0, 1.0)),
                        float(np.clip(y0 / lh, 0.0, 1.0)),
                        float(np.clip(x1 / lw, 0.0, 1.0)),
                        float(np.clip(y1 / lh, 0.0, 1.0)),
                    )
                )
                self.close()
        self._start = None

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setPen(QColor("#ffe066"))
        p.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        p.drawText(QRectF(0, 24, self.width(), 40), Qt.AlignmentFlag.AlignCenter, "Drag a box around your WoW Minimap  (ESC to cancel)")
        if self._start is not None and self._cur is not None:
            p.setPen(QPen(QColor("#ffe066"), 3.0))
            p.drawRect(QRectF(self._start, self._cur).normalized())
        p.end()


class QtRaidSyncDialog(QDialog):
    """Host or Join a ForeverNavigator session with granular opt-in sync controls."""

    def __init__(self, win: "ForeverNavigatorQtWindow"):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("ForeverNavigator — Host / Join Raid Sync")
        self.setFixedWidth(460)
        lay = QVBoxLayout(self)

        # Status & LAN IP banner
        banner = QFrame()
        banner.setObjectName("Surface")
        bl = QVBoxLayout(banner)
        bl.addWidget(QLabel(win.get_network_status_summary()))
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("Your Local LAN IP:"))
        ip_edit = QLineEdit(win.get_local_lan_ip())
        ip_edit.setReadOnly(True)
        ip_row.addWidget(ip_edit)
        bl.addLayout(ip_row)
        lay.addWidget(banner)

        # Identity
        id_row = QHBoxLayout()
        self.ed_name = QLineEdit(win.identity.name)
        self.cb_class = QComboBox()
        self.cb_class.addItems(list(WOW_CLASS_COLORS.keys()))
        self.cb_class.setCurrentText(win.identity.wow_class)
        self.cb_role = QComboBox()
        self.cb_role.addItems(["LEADER", "OFFICER", "TANK", "HEALER", "DPS"])
        self.cb_role.setCurrentText(win.identity.role)
        id_row.addWidget(self.ed_name, 2)
        id_row.addWidget(self.cb_class, 1)
        id_row.addWidget(self.cb_role, 1)
        lay.addWidget(QLabel("Player Identity:"))
        lay.addLayout(id_row)

        # Opt-in prefs
        self.chk_pos = QCheckBox("Share Live Position with Raid")
        self.chk_pos.setChecked(win.sync_prefs.sync_position)
        self.chk_vid = QCheckBox("Opt-In Video Stream / Buffer")
        self.chk_vid.setChecked(win.sync_prefs.sync_video)
        self.chk_cl = QCheckBox("Share Combat Log (after /reload or raid)")
        self.chk_cl.setChecked(win.sync_prefs.sync_combatlog)
        lay.addWidget(self.chk_pos)
        lay.addWidget(self.chk_vid)
        lay.addWidget(self.chk_cl)

        # Connection
        c_row = QHBoxLayout()
        self.ed_host = QLineEdit("127.0.0.1")
        self.ed_port = QLineEdit("8765")
        self.ed_port.setFixedWidth(70)
        c_row.addWidget(QLabel("Host/IP:"))
        c_row.addWidget(self.ed_host, 1)
        c_row.addWidget(QLabel("Port:"))
        c_row.addWidget(self.ed_port)
        lay.addLayout(c_row)

        p_row = QHBoxLayout()
        self.ed_pass = QLineEdit()
        self.ed_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_officer = QLineEdit()
        self.ed_officer.setEchoMode(QLineEdit.EchoMode.Password)
        p_row.addWidget(QLabel("Passphrase:"))
        p_row.addWidget(self.ed_pass)
        p_row.addWidget(QLabel("Officer Key:"))
        p_row.addWidget(self.ed_officer)
        lay.addLayout(p_row)

        wb_row = QHBoxLayout()
        self.cb_wb = QComboBox()
        self.cb_wb.addItems(["all", "officers", "leader"])
        self.cb_wb.setCurrentText(win.host_whiteboard_perm)
        wb_row.addWidget(QLabel("Host Whiteboard Permission:"))
        wb_row.addWidget(self.cb_wb)
        lay.addLayout(wb_row)

        self.lbl_err = QLabel("")
        self.lbl_err.setStyleSheet("color: #f85149;")
        lay.addWidget(self.lbl_err)

        btns = QHBoxLayout()
        b_host = QPushButton("Host Session")
        b_host.setObjectName("Success")
        b_host.clicked.connect(self._host)
        b_join = QPushButton("Join Session")
        b_join.setObjectName("Primary")
        b_join.clicked.connect(self._join)
        b_disc = QPushButton("Disconnect")
        b_disc.setObjectName("Danger")
        b_disc.clicked.connect(self._disconnect)
        btns.addWidget(b_host)
        btns.addWidget(b_join)
        btns.addWidget(b_disc)
        lay.addLayout(btns)

    def _save(self) -> None:
        self.win.identity.name = self.ed_name.text().strip() or "Raider"
        self.win.identity.wow_class = self.cb_class.currentText()
        self.win.identity.role = self.cb_role.currentText()
        self.win.host_whiteboard_perm = self.cb_wb.currentText()
        self.win.sync_prefs = SyncPreferences(
            sync_position=self.chk_pos.isChecked(),
            poll_hz=self.win.fps,
            sync_video=self.chk_vid.isChecked(),
            sync_combatlog=self.chk_cl.isChecked(),
        )

    def _parse_port(self) -> Optional[int]:
        try:
            port = int(self.ed_port.text().strip())
            if not (1 <= port <= 65535):
                raise ValueError
            return port
        except ValueError:
            self.lbl_err.setText("Invalid port (must be 1-65535).")
            return None

    def _host(self) -> None:
        port = self._parse_port()
        if port is None:
            return
        self._save()
        if not self.win.start_host(port, self.ed_pass.text(), self.cb_wb.currentText(), self.ed_officer.text()):
            self.lbl_err.setText(self.win.last_net_error or "Failed to start Host server.")
            return
        self.accept()

    def _join(self) -> None:
        port = self._parse_port()
        if port is None:
            return
        raw = self.ed_host.text().strip()
        if not raw:
            self.lbl_err.setText("Enter Host IP or ws:// URL.")
            return
        self._save()
        url = raw if raw.startswith(("ws://", "wss://")) else (f"ws://{raw}" if ":" in raw else f"ws://{raw}:{port}")
        if not self.win.start_client(url, self.ed_pass.text(), self.ed_officer.text()):
            self.lbl_err.setText(self.win.last_net_error or "Failed to connect.")
            return
        self.accept()

    def _disconnect(self) -> None:
        self.win.stop_network()
        self.accept()


class QtCombatStatsDialog(QDialog):
    """Loads and displays WoWCombatLog statistics (DPS, HPS, Overheal, Deaths)."""

    def __init__(self, parent: QWidget, default_logs_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("ForeverNavigator — Combat Log Analytics")
        self.resize(680, 500)
        self.default_logs_dir = default_logs_dir
        lay = QVBoxLayout(self)
        btn = QPushButton("Load WoWCombatLog-*.txt...")
        btn.setObjectName("Primary")
        btn.clicked.connect(self._load_log)
        lay.addWidget(btn)
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QFont("Consolas", 9))
        self.txt.setText("Click 'Load WoWCombatLog-*.txt...' to analyze DPS, HPS, Overhealing %, Interrupts, Dispels & Death Recaps.")
        lay.addWidget(self.txt, 1)

    def _load_log(self) -> None:
        init_dir = str(self.default_logs_dir) if self.default_logs_dir and self.default_logs_dir.exists() else "."
        path, _ = QFileDialog.getOpenFileName(self, "Select WoWCombatLog file", init_dir, "Text logs (*.txt)")
        if not path:
            return
        summary = analyze_combat_events(iter_combatlog_file(path), session_name=Path(path).name)
        lines = [f"=== {summary.name} ({summary.duration_sec:.1f}s) ===", "", "--- Top DPS ---"]
        for name, dps, total in summary.dps_leaderboard()[:15]:
            p = summary.players[name]
            lines.append(f"  {name:18s} | {dps:8.1f} DPS | Total: {total:10,d} | Int: {p.interrupts} | Deaths: {p.deaths}")
        lines.append("\n--- Top HPS ---")
        for name, hps, ovh in summary.hps_leaderboard()[:10]:
            p = summary.players[name]
            lines.append(f"  {name:18s} | {hps:8.1f} HPS | Overheal: {ovh:4.1f}% | Dispels: {p.dispels}")
        lines.append(f"\n--- Death Recaps ({len(summary.death_recaps)}) ---")
        for dr in summary.death_recaps[:10]:
            last_hit = dr.recent_events[-1] if dr.recent_events else None
            hit_txt = f"{last_hit.spell_name} ({last_hit.amount})" if last_hit else "unknown"
            lines.append(f"  {dr.player_name:18s} | Killing/Recent event: {hit_txt}")
        self.txt.setText("\n".join(lines))


class ForeverNavigatorQtWindow(QMainWindow):
    """GPU-accelerated PySide6 + ModernGL main window for ForeverNavigator."""

    def __init__(self, index_dir: str | Path, fps: Optional[float] = None):
        super().__init__()
        self.index_dir = Path(index_dir)
        self.data_dir = self.index_dir.parent
        self.roi_path = self.data_dir / "minimap_roi.json"
        self.settings_path = self.data_dir / "settings.json"
        self.settings: AppSettings = load_settings(self.settings_path)
        if fps is not None:
            self.settings.fps = float(fps)

        self._detected_storage: str = detect_storage_type(self.data_dir)
        self._effective_storage_mode: str = resolve_storage_mode(self.settings.storage_mode, self.data_dir)
        self.wow_dir = find_wow_dir(self.settings.wow_dir or None)
        self.wow_logs_dir = (self.wow_dir / "_forever_" / "Logs") if self.wow_dir else None

        self.tracker_3d = LightweightDungeonStore(index_dir, max_preview_px=512)
        self.tracker_mm = MinimapTracker(
            dungeons=self.tracker_3d.dungeons,
            dungeon_order=self.tracker_3d.dungeon_order,
            roi=load_roi(self.roi_path),
            rotate_minimap=self.settings.rotate_minimap,
            high_precision=self.settings.high_precision,
            lazy_precompute=True,
            neural_model_name=self.settings.neural_model_name or None,
            use_opencv=self.settings.use_opencv_refine,
            opencv_verify_only=self.settings.opencv_verify_only,
            use_verifier_neural=self.settings.use_verifier_neural,
            neural_max_hz=self.settings.neural_max_hz,
            verifier_hz=self.settings.verifier_hz,
        )
        self._capture_backend: str = "init"
        self._capture_ms_ema: float = 0.0
        self._last_tel_ui_t: float = 0.0
        self.viewer_3d = GPU3DViewer(wow_dir=self.wow_dir, listfile_path=self.data_dir / "listfile.csv", product="wowf")
        try:
            from ..navigator.world_maps import WorldMapRegistry

            self.world_maps: Optional[Any] = WorldMapRegistry(
                index_dir=self.index_dir,
                cache_dir=self.data_dir / "cache" / "maps",
                wow_dir=self.wow_dir,
            )
            self.world_maps.set_storage_mode(self._effective_storage_mode)
        except Exception:
            self.world_maps = None

        # Subsystems
        self.identity = PlayerIdentity(name="Player", wow_class="WARRIOR", role="DPS")
        self.sync_prefs = SyncPreferences(poll_hz=self.settings.fps)
        self.server: Optional[ForeverNavigatorServer] = None
        self.client: Optional[ForeverNavigatorClient] = None
        self.raid_players: Dict[str, PlayerState] = {}
        self.host_whiteboard_perm: str = "all"
        self.last_net_error: Optional[str] = None
        self._net_connected: bool = False
        self._ui_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._closing: bool = False
        self.whiteboard = TacticalWhiteboard()
        self.active_ruler: Optional[RulerMeasurement] = None
        self._active_stroke_pts: List[Tuple[float, float]] = []

        self.session_store = SessionStore(self.data_dir / "sessions" / f"session_{int(time.time())}.db")
        self.dev_logger = DevGameplayLogger(self.data_dir / "dev_gameplay")
        self.timeline = TimelineController(self.session_store)
        self.replay_mode: bool = False
        self.replay_players: Dict[str, PlayerState] = {}

        self.fps = self.settings.fps
        self.running = False
        self._stop_capture_event = threading.Event()
        self._selected_monitor: int = max(1, int(self.settings.monitor))
        self._latest_est: Optional[TrackerEstimate] = None
        self._last_rendered_est: Optional[TrackerEstimate] = None
        self._lock = threading.Lock()

        # Viewport & streaming state
        self.zoom: float = 1.0
        self.pan_x: float = 512.0
        self.pan_y: float = 512.0
        self.view_3d_mode: bool = False
        self._showing_azeroth_overview: bool = True
        self._azeroth_overview: Optional[Any] = None
        self._overview_zoom: float = 1.0
        self._overview_pan_x: float = 1920.0
        self._overview_pan_y: float = 1080.0
        self._is_interacting: bool = False
        self._drag_last: Optional[Tuple[float, float]] = None
        self._rdrag_last: Optional[Tuple[float, float]] = None
        self._tool_name: str = "Pan"

        self._streaming_zones: set[str] = set()
        self._fullres_uploaded_zones: set[str] = set()
        self._cont_l1_loading: set[str] = set()
        self._cont_l1_native: Dict[str, int] = {}
        self._cont_meta: Dict[str, Dict[str, Any]] = {}
        self._deep_tile_key: Optional[tuple] = None
        self._deep_tile_req_id: int = 0
        self._last_roster_sig: Optional[tuple] = None

        self._zoom_target: Optional[float] = None
        self._zoom_anchor: Optional[Tuple[float, float]] = None
        self._zoom_last_t: float = 0.0
        self._last_zoom_lbl: str = "100%"
        self._transition_state: Optional[Dict[str, Any]] = None
        self._all_map_entries: List[Any] = []
        self._filtered_map_entries: List[Any] = []
        self._roi_overlay: Optional[QtROIPickerOverlay] = None

        self.setWindowTitle("ForeverNavigator — WoW: Forever Tactical Navigator & Replay (GPU)")
        self.resize(1240, 860)
        self.setMinimumSize(840, 580)
        self._build_ui()
        self._refresh_map_catalog()

        # Timers
        self._queue_timer = QTimer(self)
        self._queue_timer.timeout.connect(self._drain_ui_queue)
        self._queue_timer.start(30)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_tracker)

        self._zoom_timer = QTimer(self)
        self._zoom_timer.setInterval(8)  # vsync-paced smooth 60-144 FPS zoom
        self._zoom_timer.timeout.connect(self._step_smooth_zoom)

        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.timeout.connect(self._on_interaction_settled)

        self._trans_timer = QTimer(self)
        self._trans_timer.setInterval(12)
        self._trans_timer.timeout.connect(self._step_fix_transition)

        self._load_azeroth_overview_async()
        self._preload_cached_maps()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Top Bar ---
        topbar = QFrame()
        topbar.setObjectName("Surface")
        topbar.setFixedHeight(44)
        tl = QHBoxLayout(topbar)
        tl.setContentsMargins(12, 4, 12, 4)
        brand = QLabel("FOREVERNAVIGATOR")
        brand.setStyleSheet("color: #f0b429; font-weight: bold; font-size: 10pt;")
        self.lbl_dungeon = QLabel("Azeroth Overview — Click 'Start Capture' or select a map")
        self.lbl_dungeon.setStyleSheet("color: #f0f6fc; font-weight: bold; font-size: 10pt;")
        tl.addWidget(brand)
        tl.addSpacing(8)
        tl.addWidget(self.lbl_dungeon, 1)

        self.pill_sync = QPushButton("Offline Solo")
        self.pill_sync.setObjectName("Pill")
        self.pill_sync.setStyleSheet("color: #58a6ff; background: #21262d; border: 1px solid #30363d; padding: 3px 8px;")
        self.pill_sync.clicked.connect(lambda: QtRaidSyncDialog(self).exec())
        self.pill_pos = QLabel("Pos: --")
        self.pill_pos.setObjectName("Pill")
        self.pill_fps = QLabel(f"{self.fps:.1f} Hz")
        self.pill_fps.setObjectName("Pill")
        tl.addWidget(self.pill_sync)
        tl.addWidget(self.pill_pos)
        tl.addWidget(self.pill_fps)
        outer.addWidget(topbar)

        # --- Body: Sidebar + GPU Viewport ---
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Surface")
        sidebar.setFixedWidth(306)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        sl.addWidget(tabs)

        # === TAB 1: MAP & NAV ===
        tab_nav = QWidget()
        tn_lay = QVBoxLayout(tab_nav)
        tn_lay.setContentsMargins(4, 6, 4, 4)
        tn_lay.setSpacing(6)

        c_cap = QFrame()
        c_cap.setObjectName("Card")
        cl = QVBoxLayout(c_cap)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(6)
        t1 = QLabel("CAPTURE & VIEW")
        t1.setObjectName("CardTitle")
        cl.addWidget(t1)

        cap_row = QHBoxLayout()
        self.btn_toggle = QPushButton("Start Capture")
        self.btn_toggle.setObjectName("Primary")
        self.btn_toggle.clicked.connect(self._toggle_capture)
        btn_roi = QPushButton("Pick Minimap...")
        btn_roi.clicked.connect(self._pick_minimap_box)
        cap_row.addWidget(self.btn_toggle, 1)
        cap_row.addWidget(btn_roi)
        cl.addLayout(cap_row)

        opt_row = QHBoxLayout()
        with mss.MSS() as sct:
            n_mons = max(1, len(sct.monitors) - 1)
        self._selected_monitor = min(self.settings.monitor, n_mons)
        opt_row.addWidget(QLabel("Mon:"))
        self.cb_mon = QComboBox()
        self.cb_mon.addItems([str(i) for i in range(1, n_mons + 1)])
        self.cb_mon.setCurrentText(str(self._selected_monitor))
        self.cb_mon.setFixedWidth(48)
        self.cb_mon.currentTextChanged.connect(lambda v: setattr(self, "_selected_monitor", max(1, int(v))))
        opt_row.addWidget(self.cb_mon)

        self.chk_follow = QCheckBox("Center Player")
        self.chk_follow.setChecked(bool(self.settings.follow_player))
        self.chk_follow.setStyleSheet("color: #f0b429;")
        self.chk_follow.toggled.connect(self._on_follow_toggle)
        opt_row.addWidget(self.chk_follow)

        self.chk_spawns = QCheckBox("Spawns")
        self.chk_spawns.setChecked(bool(self.settings.show_spawns))
        self.chk_spawns.setStyleSheet("color: #ffa53c;")
        self.chk_spawns.toggled.connect(lambda _v: self.viewport.update())
        opt_row.addWidget(self.chk_spawns)
        cl.addLayout(opt_row)

        opt2 = QHBoxLayout()
        self.chk_neural = QCheckBox("AI")
        has_nn = getattr(self.tracker_mm, "_neural", None) is not None
        self.chk_neural.setChecked(bool(self.settings.use_neural_model) and has_nn)
        self.chk_neural.setEnabled(has_nn)
        self.chk_neural.setToolTip(
            "Enable trained ONNX Neural AI localizer"
            if has_nn
            else "No ONNX model found in data/models/"
        )
        self.chk_neural.setStyleSheet("color: #3fb950;")
        self.chk_neural.toggled.connect(self._toggle_neural_model)
        self.tracker_mm.set_use_neural(self.chk_neural.isChecked())
        opt2.addWidget(self.chk_neural)

        self.chk_opencv = QCheckBox("ImgRec")
        self.chk_opencv.setChecked(bool(self.settings.use_opencv_refine))
        self.chk_opencv.setToolTip("Enable OpenCV template-matching image recognition (uncheck for Pure AI only)")
        self.chk_opencv.setStyleSheet("color: #f0b429;")
        self.chk_opencv.toggled.connect(self._toggle_opencv_refine)
        self.tracker_mm.set_use_opencv(self.chk_opencv.isChecked())
        opt2.addWidget(self.chk_opencv)

        self.chk_anchor = QCheckBox("Anchor")
        self.chk_anchor.setChecked(bool(self.settings.periodic_opencv_anchor))
        self.chk_anchor.setToolTip("Periodic full-map OpenCV geometric anchor check (~every 2.5s) to prevent drift")
        self.chk_anchor.setStyleSheet("color: #d2a8ff;")
        self.chk_anchor.toggled.connect(self._toggle_periodic_anchor)
        self.tracker_mm.set_periodic_anchor(self.chk_anchor.isChecked())
        opt2.addWidget(self.chk_anchor)

        self.chk_hiprec = QCheckBox("Hi-Prec")
        self.chk_hiprec.setChecked(bool(self.settings.high_precision))
        self.chk_hiprec.setStyleSheet("color: #58a6ff;")
        self.chk_hiprec.toggled.connect(self._toggle_high_precision)
        opt2.addWidget(self.chk_hiprec)
        cl.addLayout(opt2)

        opt2b = QHBoxLayout()
        opt2b.addWidget(QLabel("Storage:"))
        auto_lbl = f"Auto ({self._detected_storage.upper()})"
        self._storage_label_to_mode = {auto_lbl: "auto", "SSD (Low RAM)": "ssd", "HDD (More Cache)": "hdd"}
        mode_to_lbl = {v: k for k, v in self._storage_label_to_mode.items()}
        self.cb_storage = QComboBox()
        self.cb_storage.addItems(list(self._storage_label_to_mode.keys()))
        self.cb_storage.setCurrentText(mode_to_lbl.get(self.settings.storage_mode, auto_lbl))
        self.cb_storage.currentTextChanged.connect(self._on_storage_mode_change)
        opt2b.addWidget(self.cb_storage, 1)
        cl.addLayout(opt2b)

        opt3 = QHBoxLayout()
        opt3.addWidget(QLabel("Poll:"))
        self._fps_presets = [
            ("1 Hz", 1.0),
            ("2 Hz", 2.0),
            ("5 Hz", 5.0),
            ("10 Hz", 10.0),
            ("15 Hz", 15.0),
            ("30 Hz", 30.0),
            ("60 Hz", 60.0),
        ]
        self._fps_label_to_val = {lbl: val for lbl, val in self._fps_presets}
        self.cb_fps = QComboBox()
        self.cb_fps.addItems([lbl for lbl, _ in self._fps_presets])
        best_lbl = min(self._fps_presets, key=lambda kv: abs(kv[1] - self.fps))[0]
        self.cb_fps.setCurrentText(best_lbl)
        self.cb_fps.setToolTip("Overall screen capture & OpenCV image-recognition target rate")
        self.cb_fps.currentTextChanged.connect(self._on_fps_change)
        opt3.addWidget(self.cb_fps, 1)

        opt3.addWidget(QLabel("AI:"))
        self._ai_hz_presets = [
            ("Sync", 0.0),
            ("30 Hz", 30.0),
            ("15 Hz", 15.0),
            ("12 Hz", 12.0),
            ("5 Hz", 5.0),
            ("2 Hz", 2.0),
            ("1 Hz", 1.0),
        ]
        self._ai_hz_label_to_val = {lbl: val for lbl, val in self._ai_hz_presets}
        self.cb_ai_hz = QComboBox()
        self.cb_ai_hz.addItems([lbl for lbl, _ in self._ai_hz_presets])
        best_ai_lbl = min(self._ai_hz_presets, key=lambda kv: abs(kv[1] - self.settings.neural_max_hz))[0]
        self.cb_ai_hz.setCurrentText(best_ai_lbl)
        self.cb_ai_hz.setToolTip("Max polling rate for the Neural AI model (Sync = run on every capture frame)")
        self.cb_ai_hz.currentTextChanged.connect(self._on_ai_hz_change)
        opt3.addWidget(self.cb_ai_hz, 1)
        opt3.addWidget(QLabel("Ver:"))
        self._ver_hz_presets = [
            ("Off", 0.0),
            ("0.5 Hz", 0.5),
            ("1 Hz", 1.0),
            ("2 Hz", 2.0),
            ("5 Hz", 5.0),
            ("10 Hz", 10.0),
        ]
        self._ver_hz_label_to_val = {lbl: val for lbl, val in self._ver_hz_presets}
        self.cb_ver_hz = QComboBox()
        self.cb_ver_hz.addItems([lbl for lbl, _ in self._ver_hz_presets])
        best_ver_lbl = min(self._ver_hz_presets, key=lambda kv: abs(kv[1] - self.settings.verifier_hz))[0]
        self.cb_ver_hz.setCurrentText(best_ver_lbl)
        self.cb_ver_hz.setToolTip("Periodic verification rate for Accurate AI and/or Classic ImgRec (always < primary AI Hz)")
        self.cb_ver_hz.currentTextChanged.connect(self._on_ver_hz_change)
        opt3.addWidget(self.cb_ver_hz, 1)
        cl.addLayout(opt3)

        opt_pipe = QHBoxLayout()
        opt_pipe.addWidget(QLabel("Pipeline:"))
        self._pipeline_presets = [
            ("Hybrid (AI + ImgRec)", "hybrid_60hz_cv"),
            ("Accurate AI + Classic Ver", "accurate_plus_classic"),
            ("Fast v2 + Accurate AI Ver", "fast_plus_accurate"),
            ("Fast v2 + Acc AI + Classic", "fast_plus_accurate_classic"),
            ("Pure Accurate AI (GPU)", "pure_accurate_ai"),
        ]
        self._pipe_label_to_key = {lbl: key for lbl, key in self._pipeline_presets}
        self._pipe_key_to_label = {key: lbl for lbl, key in self._pipeline_presets}
        self.cb_pipeline = QComboBox()
        self.cb_pipeline.addItems([lbl for lbl, _ in self._pipeline_presets])
        self.cb_pipeline.setCurrentText(self._pipe_key_to_label.get(self.settings.pipeline_preset, "Hybrid (AI + ImgRec)"))
        self.cb_pipeline.setToolTip("One-click tracking & verification architecture preset")
        self.cb_pipeline.currentTextChanged.connect(self._on_pipeline_preset_change)
        opt_pipe.addWidget(self.cb_pipeline, 1)
        cl.addLayout(opt_pipe)

        self.lbl_telemetry = QLabel("AI: 0.0 Hz (0ms) | ImgRec: 0.0 Hz (0ms) | Idle")
        self.lbl_telemetry.setStyleSheet("color: #8b949e; font-family: Consolas, monospace; font-size: 7.5pt; padding: 2px 0px;")
        cl.addWidget(self.lbl_telemetry)

        from ..navigator.neural_localizer import format_model_label, list_available_onnx_models

        opt_nn = QHBoxLayout()
        opt_nn.addWidget(QLabel("Model:"))
        self._model_label_to_stem: Dict[str, str] = {}
        self.cb_model = QComboBox()
        avail_models = list_available_onnx_models(self.data_dir / "models")
        active_stem = getattr(getattr(self.tracker_mm, "_neural", None), "model_stem", "")
        selected_lbl = ""
        for stem, _onnx_p, meta_p in avail_models:
            lbl = format_model_label(stem, meta_p)
            self._model_label_to_stem[lbl] = stem
            self.cb_model.addItem(lbl)
            if stem == active_stem:
                selected_lbl = lbl
        if not avail_models:
            self.cb_model.addItem("None found")
            self.cb_model.setEnabled(False)
        elif selected_lbl:
            self.cb_model.setCurrentText(selected_lbl)
        self.cb_model.setToolTip("Switch between available ONNX neural models in data/models/ live")
        self.cb_model.currentTextChanged.connect(self._on_neural_model_change)
        opt_nn.addWidget(self.cb_model, 1)
        cl.addLayout(opt_nn)

        self.btn_view3d = QPushButton("View: 2D Map (GPU)")
        self.btn_view3d.clicked.connect(self._toggle_3d_view)
        cl.addWidget(self.btn_view3d)
        tn_lay.addWidget(c_cap)

        # All Maps Card
        c_maps = QFrame()
        c_maps.setObjectName("Card")
        ml = QVBoxLayout(c_maps)
        ml.setContentsMargins(8, 8, 8, 8)
        ml.setSpacing(4)
        mh = QHBoxLayout()
        t2 = QLabel("ALL MAPS (WOW: FOREVER)")
        t2.setObjectName("CardTitle")
        mh.addWidget(t2, 1)
        b_idx = QPushButton("Index")
        b_idx.setStyleSheet("padding: 2px 6px; font-size: 7.5pt; color: #f0b429;")
        b_idx.clicked.connect(self._preload_all_zones)
        b_auto = QPushButton("Auto-Detect")
        b_auto.setStyleSheet("padding: 2px 6px; font-size: 7.5pt; color: #58a6ff;")
        b_auto.clicked.connect(self._set_auto_detect)
        mh.addWidget(b_idx)
        mh.addWidget(b_auto)
        ml.addLayout(mh)

        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("Search zones, raids, dungeons...")
        self.ed_search.textChanged.connect(self._filter_map_list)
        ml.addWidget(self.ed_search)

        self.cb_cat = QComboBox()
        self.cb_cat.addItems(["All", "Raids", "Dungeons", "Battlegrounds", "Continents", "Zones", "Cities", "Caves & Buildings", "Other"])
        self.cb_cat.currentTextChanged.connect(self._filter_map_list)
        ml.addWidget(self.cb_cat)

        self.map_list = QListWidget()
        self.map_list.itemDoubleClicked.connect(lambda _it: self._load_selected_map())
        ml.addWidget(self.map_list, 1)

        b_load = QPushButton("Load Selected Map")
        b_load.setObjectName("Success")
        b_load.clicked.connect(self._load_selected_map)
        ml.addWidget(b_load)
        self.lbl_map_status = QLabel("")
        self.lbl_map_status.setStyleSheet("color: #8b949e; font-size: 7.5pt;")
        ml.addWidget(self.lbl_map_status)
        tn_lay.addWidget(c_maps, 1)
        tabs.addTab(tab_nav, "Map & Nav")

        # === TAB 2: TACTICS ===
        tab_tac = QWidget()
        tt_lay = QVBoxLayout(tab_tac)
        tt_lay.setContentsMargins(4, 6, 4, 4)

        c_tools = QFrame()
        c_tools.setObjectName("Card")
        tl2 = QVBoxLayout(c_tools)
        lbl_tt = QLabel("TACTICAL TOOLS")
        lbl_tt.setObjectName("CardTitle")
        tl2.addWidget(lbl_tt)
        grid = QGridLayout()
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        tools_list = [
            ("Pan", "Pan"),
            ("Ruler (yd)", "Ruler (yd)"),
            ("Draw", "Draw"),
            ("Text", "Text"),
            ("Range 8yd", "8 yd"),
            ("Range 30yd", "30 yd"),
            ("Range 40yd", "40 yd"),
        ]
        for idx, (val, lbl) in enumerate(tools_list):
            b = QPushButton(lbl)
            b.setCheckable(True)
            if val == "Pan":
                b.setChecked(True)
            b.clicked.connect(lambda _c, v=val: setattr(self, "_tool_name", v))
            self._tool_group.addButton(b)
            grid.addWidget(b, idx // 2, idx % 2)
        tl2.addLayout(grid)
        tt_lay.addWidget(c_tools)

        c_wb = QFrame()
        c_wb.setObjectName("Card")
        wbl = QHBoxLayout(c_wb)
        b_undo = QPushButton("Undo")
        b_undo.clicked.connect(self._undo_draw)
        b_clr = QPushButton("Clear Board")
        b_clr.setStyleSheet("color: #f85149;")
        b_clr.clicked.connect(self._clear_board)
        wbl.addWidget(b_undo)
        wbl.addWidget(b_clr)
        tt_lay.addWidget(c_wb)

        c_zoom = QFrame()
        c_zoom.setObjectName("Card")
        zl = QHBoxLayout(c_zoom)
        bm = QPushButton("-")
        bm.clicked.connect(lambda: self._adjust_zoom(1 / 1.25))
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bp = QPushButton("+")
        bp.clicked.connect(lambda: self._adjust_zoom(1.25))
        bf = QPushButton("Fit")
        bf.setStyleSheet("color: #58a6ff;")
        bf.clicked.connect(self._reset_view)
        zl.addWidget(bm)
        zl.addWidget(self.lbl_zoom, 1)
        zl.addWidget(bp)
        zl.addWidget(bf)
        tt_lay.addWidget(c_zoom)
        tt_lay.addStretch(1)
        tabs.addTab(tab_tac, "Tactics")

        # === TAB 3: RAID & REPLAY ===
        tab_raid = QWidget()
        tr_lay = QVBoxLayout(tab_raid)
        tr_lay.setContentsMargins(4, 6, 4, 4)

        b_sync = QPushButton("Raid Sync... (Host / Join)")
        b_sync.setObjectName("Purple")
        b_sync.clicked.connect(lambda: QtRaidSyncDialog(self).exec())
        tr_lay.addWidget(b_sync)
        self.lbl_roster_hdr = QLabel("Guildmates (0)")
        self.lbl_roster_hdr.setObjectName("CardTitle")
        tr_lay.addWidget(self.lbl_roster_hdr)
        self.roster_tree = QTreeWidget()
        self.roster_tree.setHeaderLabels(["Raider", "Role", "Zone", "HP"])
        tr_lay.addWidget(self.roster_tree, 1)
        b_stats = QPushButton("Combat Stats...")
        b_stats.clicked.connect(lambda: QtCombatStatsDialog(self, self.wow_logs_dir).exec())
        tr_lay.addWidget(b_stats)
        tabs.addTab(tab_raid, "Raid & Replay")

        body.addWidget(sidebar)

        # Right side: GPU Viewport + Bottom Scrubber/Status
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        self.viewport = GPUMapViewport(self)
        right.addWidget(self.viewport, 1)

        bottom = QFrame()
        bottom.setObjectName("Surface")
        btl = QVBoxLayout(bottom)
        btl.setContentsMargins(10, 4, 10, 4)
        btl.setSpacing(2)

        scrub = QHBoxLayout()
        self.chk_replay = QCheckBox("Replay Scrub")
        self.chk_replay.setStyleSheet("color: #3fb950; font-weight: bold;")
        self.chk_replay.toggled.connect(self._on_replay_toggle)
        self.scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.scrub_slider.setRange(0, 1000)
        self.scrub_slider.setValue(1000)
        self.scrub_slider.valueChanged.connect(self._on_scrub)
        self.lbl_scrub = QLabel("LIVE")
        self.lbl_scrub.setFixedWidth(72)
        self.lbl_scrub.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        scrub.addWidget(self.chk_replay)
        scrub.addWidget(self.scrub_slider, 1)
        scrub.addWidget(self.lbl_scrub)
        btl.addLayout(scrub)

        self.lbl_stats = QLabel("GPU Hardware Accelerated | Offline Solo | Scroll to zoom smoothly, drag to pan")
        self.lbl_stats.setStyleSheet("color: #8b949e; font-size: 8pt;")
        btl.addWidget(self.lbl_stats)
        right.addWidget(bottom)

        body.addLayout(right, 1)
        outer.addLayout(body, 1)

    # --- Coordinate & Scale Helpers ---
    def _active_slug(self) -> str:
        return self._latest_est.dungeon_slug if self._latest_est else self.tracker_mm.active_slug

    def _eff_scale(self, cw: int, ch: int, map_h: int) -> float:
        return (min(max(64, cw), max(64, ch)) / float(max(1, map_h))) * self.zoom

    def _overview_eff_scale(self, cw: int, ch: int) -> float:
        ov_w = float(self._azeroth_overview.width) if self._azeroth_overview is not None else 3840.0
        ov_h = float(self._azeroth_overview.height) if self._azeroth_overview is not None else 2160.0
        return min(max(64, cw) / ov_w, max(64, ch) / ov_h) * self._overview_zoom

    def _canvas_to_uv(self, sx: float, sy: float) -> Tuple[float, float]:
        cw, ch = max(1, self.viewport.width()), max(1, self.viewport.height())
        map_h = self.tracker_3d.dungeons[self._active_slug()].map_size_px
        sc = self._eff_scale(cw, ch, map_h)
        mx = self.pan_x + (sx - cw * 0.5) / sc
        my = self.pan_y + (sy - ch * 0.5) / sc
        return mx / float(map_h), my / float(map_h)

    @staticmethod
    def _is_good_position_fix(est: Optional[TrackerEstimate]) -> bool:
        if est is None:
            return False
        return (est.position_confidence >= 0.42) or (
            est.dungeon_confidence >= 0.55 and est.position_confidence >= 0.35
        )

    # --- Async Streaming (4K Overview, Level-1 4K Continents, Zone PNG, Level-2 CASC Tiles) ---
    def _load_azeroth_overview_async(self) -> None:
        def _worker() -> None:
            try:
                from ..navigator.world_maps import build_azeroth_overview

                ov = build_azeroth_overview(self.world_maps, width=3840, height=2160, use_disk_cache=True)
                rgba = _to_rgba_u8(np.asarray(ov.image, dtype=np.uint8)) if ov is not None else None
                if not self._closing:
                    self._ui_queue.put(("azeroth_overview_loaded", (ov, rgba)))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="QtBuildAzerothOverview").start()

    def _ensure_continent_l1_async(self, cont_slug: str) -> None:
        l1_key = f"__l1_{cont_slug}"
        if self.world_maps is None or self.viewport.has_texture(l1_key) or cont_slug in self._cont_l1_loading:
            return
        self._cont_l1_loading.add(cont_slug)

        def _worker(slug: str = cont_slug) -> None:
            res = None
            meta = None
            try:
                raw_res = self.world_maps.load_continent_l1_array(slug, max_px=4096)
                if raw_res is not None:
                    arr_rgb, native_px = raw_res
                    res = (_to_rgba_u8(arr_rgb), native_px)
                d = self.world_maps.load_map(slug, preview_only=True, max_preview_px=512)
                meta = {
                    "map_size_px": int(d.map_size_px),
                    "map_scale": float(d.map_scale),
                    "map_origin_xy": (float(d.map_origin_xy[0]), float(d.map_origin_xy[1])),
                }
            except Exception:
                res = None
            if not self._closing:
                self._ui_queue.put(("cont_l1_loaded", (slug, res, meta)))

        threading.Thread(target=_worker, daemon=True, name=f"QtLoadL1-{cont_slug}").start()

    def _request_deep_tiles_async(
        self, cont_slug: str, cx_cont: float, cy_cont: float, sc_cont: float, cw: int, ch: int
    ) -> None:
        key = (cont_slug, int(round(cx_cont)), int(round(cy_cont)), round(sc_cont, 3), cw, ch)
        if self._deep_tile_key == key or getattr(self, "_deep_tile_inflight_key", None) == key:
            return
        self._deep_tile_req_id += 1
        req_id = self._deep_tile_req_id
        self._deep_tile_inflight_key = key

        def _worker() -> None:
            arr = None
            try:
                if req_id != self._deep_tile_req_id or self._closing:
                    return
                if self.world_maps is not None and self.world_maps.ensure_full_res_tiles(cont_slug):
                    if req_id != self._deep_tile_req_id or self._closing:
                        return
                    vp = self.world_maps.render_full_res_viewport(cont_slug, cx_cont, cy_cont, sc_cont, cw, ch)
                    if vp is not None and req_id == self._deep_tile_req_id:
                        arr = _to_rgba_u8(np.asarray(vp, dtype=np.uint8))
            except Exception:
                arr = None
            if not self._closing and req_id == self._deep_tile_req_id:
                self._ui_queue.put(("deep_tile_ready", (req_id, key, arr)))

        threading.Thread(target=_worker, daemon=True, name=f"QtDeepTile-{req_id}").start()

    def _ensure_zone_texture(self, slug: str, d: Any) -> None:
        z_key = f"zone_{slug}"
        if not self.viewport.has_texture(z_key):
            self.viewport.queue_texture_upload(z_key, d.map_image, build_mips=True)
        full_path = getattr(d, "map_full_path", None)
        if (
            full_path is not None
            and d.map_image.shape[0] < d.map_size_px
            and slug not in self._fullres_uploaded_zones
            and slug not in self._streaming_zones
        ):
            self._streaming_zones.add(slug)

            def _worker(p: Path = Path(full_path), s: str = slug) -> None:
                arr_rgba = None
                try:
                    arr_rgb = None
                    if p.suffix.lower() == ".png" and p.exists():
                        with Image.open(p) as raw:
                            arr_rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
                    elif p.exists():
                        with np.load(p, allow_pickle=False) as z:
                            arr_rgb = z["map_image"].astype(np.uint8)
                    if arr_rgb is not None:
                        if arr_rgb.shape[0] > 512:
                            self.tracker_mm.upgrade_zone_resolution(s, arr_rgb, max_work_px=1024)
                        arr_rgba = _to_rgba_u8(arr_rgb)
                except Exception:
                    arr_rgba = None
                if not self._closing:
                    self._ui_queue.put(("zone_fullres_loaded", (s, arr_rgba)))

            threading.Thread(target=_worker, daemon=True, name=f"QtStreamZone-{slug}").start()

    def _set_zoom_label(self, zoom_val: float) -> None:
        txt = f"{int(round(zoom_val * 100))}%"
        if txt != self._last_zoom_lbl:
            self._last_zoom_lbl = txt
            self.lbl_zoom.setText(txt)

    # --- Smooth Zoom & Pan Interactions ---
    def _on_mousewheel(self, event: QWheelEvent) -> None:
        delta = float(event.angleDelta().y())
        if abs(delta) < 1e-3:
            return
        factor = 1.24 ** (delta / 120.0)
        self._cancel_fix_transition()
        self._deep_tile_key = None
        if self.view_3d_mode:
            base = self._zoom_target if self._zoom_target is not None else self.viewer_3d.zoom
            self._zoom_target = max(0.4, min(6.0, base * factor))
        elif self._showing_azeroth_overview:
            base = self._zoom_target if self._zoom_target is not None else self._overview_zoom
            self._zoom_target = max(0.5, min(64.0, base * factor))
        else:
            base = self._zoom_target if self._zoom_target is not None else self.zoom
            self._zoom_target = max(0.10, min(8.0, base * factor))
        pos = event.position()
        self._zoom_anchor = (float(pos.x()), float(pos.y()))
        self._is_interacting = True
        self._settle_timer.stop()
        if not self._zoom_timer.isActive():
            self._zoom_last_t = time.perf_counter()
            self._zoom_timer.start()

    def _step_smooth_zoom(self) -> None:
        if self._zoom_target is None or self._closing:
            self._zoom_timer.stop()
            return
        now_t = time.perf_counter()
        dt = max(0.001, min(0.05, now_t - self._zoom_last_t)) if self._zoom_last_t > 0.0 else 0.008
        self._zoom_last_t = now_t
        # Frame-rate-independent crisp exponential ease (~110ms settle at any refresh rate)
        ease = 1.0 - math.exp(-dt * 24.0)

        cw, ch = max(1, self.viewport.width()), max(1, self.viewport.height())
        ax, ay = self._zoom_anchor if self._zoom_anchor is not None else (cw * 0.5, ch * 0.5)
        target = self._zoom_target

        if self.view_3d_mode:
            cur = self.viewer_3d.zoom
            nxt = target if abs(target / max(1e-5, cur) - 1.0) < 0.004 else cur * ((target / cur) ** ease)
            self.viewer_3d.zoom = nxt
            self._set_zoom_label(nxt)
            self._render_3d_frame()
            done = (nxt == target)
        elif self._showing_azeroth_overview:
            cur = self._overview_zoom
            old_sc = self._overview_eff_scale(cw, ch)
            nxt = target if abs(target / max(1e-5, cur) - 1.0) < 0.003 else cur * ((target / cur) ** ease)
            self._overview_zoom = nxt
            new_sc = self._overview_eff_scale(cw, ch)
            if self._azeroth_overview is not None and old_sc > 1e-6 and new_sc > 1e-6:
                wx_a = self._overview_pan_x + (ax - cw * 0.5) / old_sc
                wy_a = self._overview_pan_y + (ay - ch * 0.5) / old_sc
                self._overview_pan_x = max(0.0, min(float(self._azeroth_overview.width), wx_a - (ax - cw * 0.5) / new_sc))
                self._overview_pan_y = max(0.0, min(float(self._azeroth_overview.height), wy_a - (ay - ch * 0.5) / new_sc))
            self._set_zoom_label(nxt)
            self.viewport.update()
            done = (nxt == target)
        else:
            d = self.tracker_3d.dungeons[self._active_slug()]
            map_h = d.map_size_px
            cur = self.zoom
            old_sc = self._eff_scale(cw, ch, map_h)
            nxt = target if abs(target / max(1e-5, cur) - 1.0) < 0.003 else cur * ((target / cur) ** ease)
            self.zoom = nxt
            new_sc = self._eff_scale(cw, ch, map_h)
            if not self.chk_follow.isChecked() and old_sc > 1e-6 and new_sc > 1e-6:
                mx_a = self.pan_x + (ax - cw * 0.5) / old_sc
                my_a = self.pan_y + (ay - ch * 0.5) / old_sc
                self.pan_x = max(0.0, min(float(map_h), mx_a - (ax - cw * 0.5) / new_sc))
                self.pan_y = max(0.0, min(float(map_h), my_a - (ay - ch * 0.5) / new_sc))
            self._set_zoom_label(nxt)
            self.viewport.update()
            done = (nxt == target)

        if done:
            self._zoom_timer.stop()
            self._zoom_target = None
            self._zoom_anchor = None
            self._settle_timer.start(80)

    def _on_interaction_settled(self) -> None:
        self._is_interacting = False
        self.viewport.update()

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        pos = event.position()
        if event.button() == Qt.MouseButton.RightButton:
            self._rdrag_last = (pos.x(), pos.y())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_last = (pos.x(), pos.y())
        if self.view_3d_mode or self._showing_azeroth_overview:
            return
        slug = self._active_slug()
        uv = self._canvas_to_uv(pos.x(), pos.y())
        if self._tool_name == "Ruler (yd)":
            self.active_ruler = RulerMeasurement(uv, uv, 0.0)
        elif self._tool_name == "Draw":
            if not self._can_edit_whiteboard():
                self.lbl_stats.setText("Whiteboard editing restricted by Host.")
                self._active_stroke_pts = []
                return
            self._active_stroke_pts = [uv]
        elif self._tool_name == "Text":
            if not self._can_edit_whiteboard():
                self.lbl_stats.setText("Whiteboard editing restricted by Host.")
                return
            txt, ok = QInputDialog.getText(self, "Map Annotation", "Enter tactical note:")
            if ok and txt:
                op = WhiteboardOp(author_id=self.identity.player_id, dungeon_slug=slug, kind="text", points=[uv], text=txt, color="#ffe066")
                self.whiteboard.add_op(op)
                self._broadcast_draw(op.to_dict())
                self.viewport.update()
        elif self._tool_name.startswith("Range "):
            if not self._can_edit_whiteboard():
                self.lbl_stats.setText("Whiteboard editing restricted by Host.")
                return
            yd = float(self._tool_name.replace("Range ", "").replace("yd", ""))
            col = "#ed8936" if yd <= 10 else ("#4299e1" if yd <= 30 else "#48bb78")
            op = WhiteboardOp(author_id=self.identity.player_id, dungeon_slug=slug, kind="circle_range", points=[uv], radius_yd=yd, color=col)
            self.whiteboard.add_op(op)
            self._broadcast_draw(op.to_dict())
            self.viewport.update()

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._rdrag_last is not None and self.view_3d_mode:
            dx, dy = pos.x() - self._rdrag_last[0], pos.y() - self._rdrag_last[1]
            self._rdrag_last = (pos.x(), pos.y())
            if self.chk_follow.isChecked():
                self.chk_follow.setChecked(False)
            self.viewer_3d.pan(dx, dy, self._active_slug())
            self._render_3d_frame()
            return

        if self._drag_last is None:
            return
        dx, dy = pos.x() - self._drag_last[0], pos.y() - self._drag_last[1]
        self._drag_last = (pos.x(), pos.y())

        if self.view_3d_mode:
            self.viewer_3d.orbit(dx, dy)
            self._render_3d_frame()
            return

        if self._showing_azeroth_overview:
            self._cancel_fix_transition()
            self._zoom_timer.stop()
            self._zoom_target = None
            self._deep_tile_key = None
            if self._azeroth_overview is not None:
                cw, ch = max(1, self.viewport.width()), max(1, self.viewport.height())
                sc_ov = self._overview_eff_scale(cw, ch)
                self._overview_pan_x = max(0.0, min(float(self._azeroth_overview.width), self._overview_pan_x - dx / sc_ov))
                self._overview_pan_y = max(0.0, min(float(self._azeroth_overview.height), self._overview_pan_y - dy / sc_ov))
                self._is_interacting = True
                self._settle_timer.start(100)
                self.viewport.update()
            return

        d = self.tracker_3d.dungeons[self._active_slug()]
        map_h = d.map_size_px
        if self._tool_name == "Pan":
            if self.chk_follow.isChecked():
                self.chk_follow.setChecked(False)
            cw, ch = max(1, self.viewport.width()), max(1, self.viewport.height())
            sc = self._eff_scale(cw, ch, map_h)
            self.pan_x = max(0.0, min(float(map_h), self.pan_x - dx / sc))
            self.pan_y = max(0.0, min(float(map_h), self.pan_y - dy / sc))
            self.viewport.update()
        elif self._tool_name == "Ruler (yd)" and self.active_ruler is not None:
            uv = self._canvas_to_uv(pos.x(), pos.y())
            yd = measure_yards(*self.active_ruler.start_uv, *uv, map_h, d.map_scale)
            self.active_ruler = RulerMeasurement(self.active_ruler.start_uv, uv, yd)
            self.viewport.update()
        elif self._tool_name == "Draw" and self._active_stroke_pts:
            self._active_stroke_pts.append(self._canvas_to_uv(pos.x(), pos.y()))
            self.viewport.update()

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._rdrag_last = None
            return
        self._drag_last = None
        if self._tool_name == "Draw" and len(self._active_stroke_pts) >= 2:
            if self._can_edit_whiteboard():
                op = WhiteboardOp(
                    author_id=self.identity.player_id,
                    dungeon_slug=self._active_slug(),
                    kind="stroke",
                    points=list(self._active_stroke_pts),
                    color="#ffe066",
                    width=3,
                )
                self.whiteboard.add_op(op)
                self._broadcast_draw(op.to_dict())
            self._active_stroke_pts.clear()
            self.viewport.update()

    # --- Smooth 2-Phase Azeroth -> Fix Transition ---
    def _cancel_fix_transition(self) -> None:
        self._trans_timer.stop()
        self._transition_state = None

    def _start_fix_transition(self, est: TrackerEstimate) -> None:
        self._cancel_fix_transition()
        d = self.tracker_3d.dungeons.get(est.dungeon_slug)
        if d is None:
            self._showing_azeroth_overview = False
            self.viewport.update()
            return
        self._ensure_zone_texture(est.dungeon_slug, d)
        if self._azeroth_overview is None:
            self._showing_azeroth_overview = False
            self.pan_x, self.pan_y = float(est.map_px[0]), float(est.map_px[1])
            self.zoom = 1.4
            self.viewport.update()
            return
        entry = self.world_maps.get_entry(est.dungeon_slug) if self.world_maps is not None else None
        folder = entry.minimap_folder if entry is not None else None
        tx_ov, ty_ov = self._azeroth_overview.world_to_overview_px(folder, float(est.world_pos[0]), float(est.world_pos[1]))
        self._transition_state = {
            "t0": time.perf_counter(),
            "duration": 0.72,
            "start_pan": (self._overview_pan_x, self._overview_pan_y),
            "target_ov": (tx_ov, ty_ov),
            "start_zoom": self._overview_zoom,
            "est": est,
        }
        self._is_interacting = True
        self.lbl_dungeon.setText(f"Position Fix: {est.dungeon_name} — Zooming in...")
        self._trans_timer.start()

    def _step_fix_transition(self) -> None:
        st = self._transition_state
        if st is None or self._closing:
            self._trans_timer.stop()
            return
        t = min(1.0, (time.perf_counter() - st["t0"]) / st["duration"])
        est: TrackerEstimate = self._latest_est or st["est"]
        split = 0.62
        if t < split:
            u = t / split
            ease = 1.0 - (1.0 - u) ** 3
            sx, sy = st["start_pan"]
            tx, ty = st["target_ov"]
            self._overview_pan_x = sx + (tx - sx) * ease
            self._overview_pan_y = sy + (ty - sy) * ease
            self._overview_zoom = st["start_zoom"] * (7.0 ** ease)
            self._showing_azeroth_overview = True
        else:
            u = (t - split) / max(1e-4, 1.0 - split)
            ease = 1.0 - (1.0 - u) ** 3
            self._showing_azeroth_overview = False
            self.pan_x = float(est.map_px[0])
            self.pan_y = float(est.map_px[1])
            self.zoom = 0.65 + (1.45 - 0.65) * ease
            self.lbl_zoom.setText(f"{int(round(self.zoom * 100))}%")
        self.viewport.update()
        if t >= 1.0:
            self._trans_timer.stop()
            self._transition_state = None
            self._is_interacting = False
            self.lbl_dungeon.setText(f"{est.dungeon_name}  [{est.dungeon_confidence * 100:.0f}% | {len(self.raid_players)} synced]")
            self.viewport.update()

    # --- UI Controls & Actions ---
    def _on_follow_toggle(self, enabled: bool) -> None:
        self.settings.follow_player = bool(enabled)
        save_settings(self.settings_path, self.settings)
        if enabled and self._latest_est is not None and not self.replay_mode:
            self.pan_x = float(self._latest_est.map_px[0])
            self.pan_y = float(self._latest_est.map_px[1])
            self.viewport.update()

    def _on_storage_mode_change(self, lbl: str) -> None:
        mode = self._storage_label_to_mode.get(lbl, "auto")
        self.settings.storage_mode = mode
        save_settings(self.settings_path, self.settings)
        self._effective_storage_mode = resolve_storage_mode(mode, self.data_dir)
        if self.world_maps is not None and hasattr(self.world_maps, "set_storage_mode"):
            self.world_maps.set_storage_mode(self._effective_storage_mode)
        self.lbl_stats.setText(f"Storage streaming: {self._effective_storage_mode.upper()} mode")

    def _on_fps_change(self, lbl: str) -> None:
        val = float(self._fps_label_to_val.get(lbl, 5.0))
        self.fps = val
        self.settings.fps = val
        save_settings(self.settings_path, self.settings)
        self.pill_fps.setText(f"{val:.0f} Hz" if val >= 5.0 else f"{val:.1f} Hz")
        # Keep the visual glide timer at 60 FPS (16ms) regardless of capture polling Hz
        # so 5 Hz polling still glides smoothly in the viewport!
        if self.running and not self._poll_timer.isActive():
            self._poll_timer.start(16)
        self.lbl_stats.setText(f"Tracking polling rate: {lbl}")

    def _toggle_neural_model(self, enabled: bool) -> None:
        self.settings.use_neural_model = bool(enabled)
        save_settings(self.settings_path, self.settings)
        self.tracker_mm.set_use_neural(bool(enabled))
        self.lbl_stats.setText(f"Tracker engine: {'Neural ONNX + OpenCV Hybrid' if enabled else 'Classic OpenCV Only'}")

    def _toggle_periodic_anchor(self, enabled: bool) -> None:
        self.settings.periodic_opencv_anchor = bool(enabled)
        save_settings(self.settings_path, self.settings)
        self.tracker_mm.set_periodic_anchor(bool(enabled))
        self.lbl_stats.setText(f"Periodic OpenCV Anchor: {'ON (~2.5s)' if enabled else 'OFF (Max FPS)'}")

    def _toggle_opencv_refine(self, enabled: bool) -> None:
        self.settings.use_opencv_refine = bool(enabled)
        save_settings(self.settings_path, self.settings)
        self.tracker_mm.set_use_opencv(bool(enabled))
        self.lbl_stats.setText(f"Image Recognition (OpenCV): {'ON' if enabled else 'OFF (Pure AI)'}")

    def _on_ai_hz_change(self, label: str) -> None:
        val = self._ai_hz_label_to_val.get(label, 0.0)
        self.settings.neural_max_hz = float(val)
        save_settings(self.settings_path, self.settings)
        self.tracker_mm.set_neural_max_hz(float(val))
        self.lbl_stats.setText(f"AI Model Rate Cap: {'Match Poll' if val <= 0 else f'{val:.0f} Hz'}")

    def _on_ver_hz_change(self, label: str) -> None:
        val = self._ver_hz_label_to_val.get(label, 2.0)
        self.settings.verifier_hz = float(val)
        save_settings(self.settings_path, self.settings)
        self.tracker_mm.set_verifier_hz(float(val))
        self.lbl_stats.setText(f"Verifier Rate: {'OFF' if val <= 0 else f'{val:g} Hz'}")

    def _on_pipeline_preset_change(self, label: str) -> None:
        key = self._pipe_label_to_key.get(label, "hybrid_60hz_cv")
        self.settings.pipeline_preset = key
        models_dir = self.data_dir / "models"
        if key == "accurate_plus_classic":
            self._select_model_stem_ui("forever_minimap_v2_accurate", models_dir)
            self.settings.use_neural_model = True
            self.settings.use_opencv_refine = True
            self.settings.opencv_verify_only = True
            self.settings.use_verifier_neural = False
            self.settings.neural_max_hz = 0.0
        elif key == "fast_plus_accurate":
            self._select_model_stem_ui("forever_minimap_v2_fast", models_dir)
            self.settings.use_neural_model = True
            self.settings.use_opencv_refine = False
            self.settings.opencv_verify_only = True
            self.settings.use_verifier_neural = True
            self.settings.neural_max_hz = 0.0
        elif key == "fast_plus_accurate_classic":
            self._select_model_stem_ui("forever_minimap_v2_fast", models_dir)
            self.settings.use_neural_model = True
            self.settings.use_opencv_refine = True
            self.settings.opencv_verify_only = True
            self.settings.use_verifier_neural = True
            self.settings.neural_max_hz = 0.0
        elif key == "pure_accurate_ai":
            self._select_model_stem_ui("forever_minimap_v2_accurate", models_dir)
            self.settings.use_neural_model = True
            self.settings.use_opencv_refine = False
            self.settings.opencv_verify_only = True
            self.settings.use_verifier_neural = False
            self.settings.neural_max_hz = 0.0
        else:  # hybrid_60hz_cv (Hybrid AI + ImgRec, synced)
            self.settings.use_neural_model = True
            self.settings.use_opencv_refine = True
            self.settings.opencv_verify_only = False
            self.settings.use_verifier_neural = False
            self.settings.neural_max_hz = 0.0

        # Apply to tracker & sync checkboxes without re-triggering cascade
        self.tracker_mm.set_use_neural(self.settings.use_neural_model)
        self.tracker_mm.set_use_opencv(self.settings.use_opencv_refine)
        self.tracker_mm.set_opencv_verify_only(self.settings.opencv_verify_only)
        self.tracker_mm.set_use_verifier_neural(self.settings.use_verifier_neural, models_dir=models_dir)
        self.tracker_mm.set_neural_max_hz(self.settings.neural_max_hz)
        self.chk_neural.blockSignals(True)
        self.chk_neural.setChecked(self.settings.use_neural_model)
        self.chk_neural.blockSignals(False)
        self.chk_opencv.blockSignals(True)
        self.chk_opencv.setChecked(self.settings.use_opencv_refine)
        self.chk_opencv.blockSignals(False)
        self.cb_ai_hz.blockSignals(True)
        best_ai_lbl = min(self._ai_hz_presets, key=lambda kv: abs(kv[1] - self.settings.neural_max_hz))[0]
        self.cb_ai_hz.setCurrentText(best_ai_lbl)
        self.cb_ai_hz.blockSignals(False)
        save_settings(self.settings_path, self.settings)
        self.lbl_stats.setText(f"Pipeline -> {label}")

    def _select_model_stem_ui(self, stem: str, models_dir: Path) -> None:
        for lbl, s in self._model_label_to_stem.items():
            if s == stem:
                self.tracker_mm.set_neural_model(stem, models_dir=models_dir)
                self.settings.neural_model_name = stem
                self.cb_model.blockSignals(True)
                self.cb_model.setCurrentText(lbl)
                self.cb_model.blockSignals(False)
                return

    def _on_neural_model_change(self, label: str) -> None:
        stem = self._model_label_to_stem.get(label, "")
        if not stem:
            return
        ok = self.tracker_mm.set_neural_model(stem, models_dir=self.data_dir / "models")
        if ok:
            self.settings.neural_model_name = stem
            save_settings(self.settings_path, self.settings)
            self.chk_neural.setEnabled(True)
            self.lbl_stats.setText(f"Switched Neural Model -> {label}")
        else:
            self.lbl_stats.setText(f"Failed to load model: {stem}")

    def _toggle_high_precision(self, enabled: bool) -> None:
        self.settings.high_precision = bool(enabled)
        save_settings(self.settings_path, self.settings)
        threading.Thread(target=lambda: self.tracker_mm.set_high_precision(bool(enabled)), daemon=True).start()
        self.lbl_stats.setText(f"Tracking mode: {'High Precision (CPU+GPU)' if enabled else 'Standard'}")

    def _toggle_3d_view(self) -> None:
        self.view_3d_mode = not self.view_3d_mode
        self._scene_3d_error = None
        self.btn_view3d.setText("View: 3D Model (GPU)" if self.view_3d_mode else "View: 2D Map (GPU)")
        self.btn_view3d.setObjectName("Purple" if self.view_3d_mode else "")
        self.btn_view3d.style().unpolish(self.btn_view3d)
        self.btn_view3d.style().polish(self.btn_view3d)
        if self.view_3d_mode:
            self._render_3d_frame()
        else:
            self.viewport.update()

    def _request_3d_scene_async(self, slug: str) -> None:
        loading = getattr(self, "_scene_3d_loading", None)
        if loading == slug:
            return
        self._scene_3d_loading = slug
        self._scene_3d_error = None

        def _worker(s: str = slug) -> None:
            prepared, err = self.viewer_3d.prepare_scene_cpu(s)
            if not self._closing:
                self._ui_queue.put(("scene_3d_prepared", (s, prepared, err)))

        threading.Thread(target=_worker, daemon=True, name=f"QtLoad3D-{slug}").start()

    def _render_3d_frame(self) -> None:
        if not self.view_3d_mode:
            return
        slug = self._active_slug()
        if self.viewer_3d._loaded_slug != slug or self.viewer_3d._scene is None:
            self._request_3d_scene_async(slug)
        self.viewport.update()

    def _adjust_zoom(self, factor: float) -> None:
        self._zoom_timer.stop()
        self._zoom_target = None
        self._deep_tile_key = None
        if self.view_3d_mode:
            self.viewer_3d.zoom = float(np.clip(self.viewer_3d.zoom * factor, 0.4, 6.0))
            self.lbl_zoom.setText(f"{int(round(self.viewer_3d.zoom * 100))}%")
            self._render_3d_frame()
        elif self._showing_azeroth_overview:
            self._cancel_fix_transition()
            self._overview_zoom = float(np.clip(self._overview_zoom * factor, 0.5, 64.0))
            self.lbl_zoom.setText(f"{int(round(self._overview_zoom * 100))}%")
            self.viewport.update()
        else:
            self.zoom = float(np.clip(self.zoom * factor, 0.10, 8.0))
            self.lbl_zoom.setText(f"{int(round(self.zoom * 100))}%")
            self.viewport.update()

    def _reset_view(self) -> None:
        self._zoom_timer.stop()
        self._zoom_target = None
        self._deep_tile_key = None
        if self.view_3d_mode:
            self.viewer_3d.reset_camera(self._active_slug())
            self._render_3d_frame()
        elif self._showing_azeroth_overview:
            self._overview_zoom = 1.0
            if self._azeroth_overview is not None:
                self._overview_pan_x = self._azeroth_overview.width * 0.5
                self._overview_pan_y = self._azeroth_overview.height * 0.5
            self.viewport.update()
        else:
            self.zoom = 1.0
            half = self.tracker_3d.dungeons[self._active_slug()].map_size_px * 0.5
            self.pan_x, self.pan_y = half, half
            self.viewport.update()
        self.lbl_zoom.setText("100%")

    def _pick_minimap_box(self) -> None:
        with mss.MSS() as sct:
            mon_idx = min(self._selected_monitor, len(sct.monitors) - 1)
            mon = dict(sct.monitors[mon_idx])

        # Match the Qt QScreen corresponding to this physical MSS monitor (handles 125%/150%/200% DPI scaling)
        matched_screen = None
        qt_app = QApplication.instance()
        if qt_app is not None:
            screens = qt_app.screens()
            mw, mh = int(mon["width"]), int(mon["height"])
            for scr in screens:
                g = scr.geometry()
                dpr = float(scr.devicePixelRatio()) or 1.0
                if abs(int(round(g.width() * dpr)) - mw) <= 4 and abs(int(round(g.height() * dpr)) - mh) <= 4:
                    matched_screen = scr
                    break
            if matched_screen is None and 0 <= mon_idx - 1 < len(screens):
                matched_screen = screens[mon_idx - 1]

        def on_roi(roi: MinimapROI) -> None:
            self.tracker_mm.roi = roi
            save_roi(self.roi_path, roi)
            self.lbl_stats.setText(f"Saved Minimap ROI: ({roi.x0:.2f}, {roi.y0:.2f}) -> ({roi.x1:.2f}, {roi.y1:.2f})")

        self._roi_overlay = QtROIPickerOverlay(mon, on_roi, screen=matched_screen)
        self._roi_overlay.show()

    def _undo_draw(self) -> None:
        if not self._can_edit_whiteboard():
            return
        slug = self._active_slug()
        self.whiteboard.undo(slug)
        self._broadcast_draw({"kind": "undo", "dungeon_slug": slug})
        self.viewport.update()

    def _clear_board(self) -> None:
        if not self._can_edit_whiteboard():
            return
        slug = self._active_slug()
        self.whiteboard.clear(slug)
        self.active_ruler = None
        self._broadcast_draw({"kind": "clear_all", "dungeon_slug": slug})
        self.viewport.update()

    # --- Map Catalog ---
    def _refresh_map_catalog(self) -> None:
        if self.world_maps is not None:
            try:
                self._all_map_entries = list(self.world_maps.list_all_maps())
            except Exception:
                self._all_map_entries = []
        self._filter_map_list()

    def _filter_map_list(self) -> None:
        cat = self.cb_cat.currentText()
        q = self.ed_search.text().strip().lower()
        filtered = [
            e for e in self._all_map_entries
            if (cat == "All" or getattr(e, "category", "") == cat)
            and (not q or q in e.name.lower() or q in e.slug.lower())
        ]
        self._filtered_map_entries = filtered
        self.map_list.clear()
        self.map_list.addItems([f"[{getattr(e, 'category', 'Map')[:3].upper()}] {e.name}" for e in filtered])
        self.lbl_map_status.setText(f"{len(filtered)} / {len(self._all_map_entries)} maps")

    def _set_auto_detect(self) -> None:
        self.tracker_mm.locked_dungeon_slug = None
        self._latest_est = None
        self._cancel_fix_transition()
        self._showing_azeroth_overview = True
        self._reset_view()
        self.lbl_dungeon.setText("Azeroth Overview — Searching for Position Fix..." if self.running else "Azeroth Overview — Click 'Start Capture' or select a map")
        self.lbl_map_status.setText("Auto-Detect enabled")

    def _load_selected_map(self) -> None:
        idx = self.map_list.currentRow()
        if idx < 0 or idx >= len(self._filtered_map_entries):
            return
        entry = self._filtered_map_entries[idx]
        slug = entry.slug
        if slug in self.tracker_3d.dungeons:
            self.tracker_mm.reset()
            self.tracker_mm.locked_dungeon_slug = slug
            self.tracker_mm.active_slug = slug
            self._latest_est = None
            self._cancel_fix_transition()
            self._showing_azeroth_overview = False
            self.lbl_dungeon.setText(f"{entry.name}  [{len(self.raid_players)} synced]")
            self.lbl_map_status.setText(f"Active: {entry.name}")
            self._reset_view()
            return
        if self.world_maps is None:
            return
        self.lbl_map_status.setText(f"Loading {entry.name}...")
        self.world_maps.load_map_async(slug, lambda s, d, e: self._ui_queue.put(("map_loaded", (s, d, e))), preview_only=True)

    def _preload_cached_maps(self) -> None:
        if self.world_maps is None:
            return

        def _worker() -> None:
            for entry in list(self._all_map_entries):
                if self._closing:
                    return
                if entry.cached and entry.slug not in self.tracker_3d.dungeons and entry.category != "Continents":
                    try:
                        d = self.world_maps.load_map(entry.slug, preview_only=True)
                        if not self._closing:
                            self._ui_queue.put(("map_cached_register", (entry.slug, d)))
                    except Exception:
                        pass

        threading.Thread(target=_worker, daemon=True, name="QtPreloadCached").start()

    def _preload_all_zones(self) -> None:
        if self.world_maps is None:
            return
        self.lbl_map_status.setText("Indexing WoW: Forever zones in background...")

        def _worker() -> None:
            done = 0
            targets = [e for e in self._all_map_entries if e.category in ("Zones", "Cities", "Battlegrounds", "Dungeons", "Raids") and e.slug not in self.tracker_3d.dungeons]
            for entry in targets:
                if self._closing:
                    return
                try:
                    d = self.world_maps.load_map(entry.slug, preview_only=True)
                    done += 1
                    if not self._closing:
                        self._ui_queue.put(("map_cached_register", (entry.slug, d)))
                        if done % 5 == 0 or done == len(targets):
                            self._ui_queue.put(("log", f"Auto-Detect indexed {done}/{len(targets)} maps"))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True, name="QtIndexZones").start()

    # --- Capture Loop ---
    def _toggle_capture(self) -> None:
        if self.running:
            self.running = False
            self._stop_capture_event.set()
            self._poll_timer.stop()
            self.btn_toggle.setText("Start Capture")
            self.btn_toggle.setObjectName("Primary")
            self.btn_toggle.style().unpolish(self.btn_toggle)
            self.btn_toggle.style().polish(self.btn_toggle)
        else:
            self.running = True
            self._stop_capture_event.clear()
            self.btn_toggle.setText("Stop Capture")
            self.btn_toggle.setObjectName("Danger")
            self.btn_toggle.style().unpolish(self.btn_toggle)
            self.btn_toggle.style().polish(self.btn_toggle)
            if self._showing_azeroth_overview:
                self.lbl_dungeon.setText("Azeroth Overview — Searching for Position Fix...")
            threading.Thread(target=self._capture_loop, daemon=True, name="QtCaptureLoop").start()
            self._poll_timer.start(16)  # 60 FPS visual interpolation timer (skips repaint when stationary)

    def _capture_loop(self) -> None:
        last_fs_grab_t = 0.0
        last_db_rec_t = 0.0
        last_db_slug = ""
        cap = FastScreenCapturer(self._selected_monitor)
        self._capture_backend = cap.backend.upper()
        self._ui_queue.put(("log", f"Capture backend: {cap.backend.upper()} ({self.fps:.0f} Hz target)"))
        try:
            while self.running:
                t0 = time.perf_counter()
                interval = 1.0 / max(0.25, self.fps)
                try:
                    roi = self.tracker_mm.roi
                    mm_rgb = cap.grab_roi(self._selected_monitor, roi.x0, roi.y0, roi.x1, roi.y1)
                    capture_ms = (time.perf_counter() - t0) * 1000.0
                    self._capture_ms_ema = (
                        capture_ms if self._capture_ms_ema <= 0.0 else 0.25 * capture_ms + 0.75 * self._capture_ms_ema
                    )
                    self._capture_backend = cap.backend.upper()
                    est = self.tracker_mm.update(mm_rgb, is_already_cropped=True)
                    if self.dev_logger.enabled:
                        full_rgb = None
                        now_m = time.monotonic()
                        if (now_m - last_fs_grab_t) >= max(15.0, self.dev_logger.fullscreen_interval_sec):
                            last_fs_grab_t = now_m
                            full_rgb = cap.grab_fullscreen(self._selected_monitor)
                        self.dev_logger.log_tracker_tick(
                            est, self.tracker_mm.locked_dungeon_slug, capture_ms,
                            self.tracker_mm.last_crop_rgb, full_rgb, roi.to_dict(),
                        )
                    st = PlayerState(
                        player_id=self.identity.player_id,
                        name=self.identity.name,
                        wow_class=self.identity.wow_class,
                        role=self.identity.role,
                        dungeon_slug=est.dungeon_slug,
                        x=float(est.world_pos[0]),
                        y=float(est.world_pos[1]),
                        z=float(est.world_pos[2]),
                        map_px=est.map_px,
                        confidence=est.position_confidence,
                        t_host_ms=self.client.host_now_ms() if self.client else now_ms(),
                    )
                    # Record replay positions at <= 4 Hz (or on zone change) instead of 60 SQLite writes/sec
                    if (t0 - last_db_rec_t) >= 0.25 or est.dungeon_slug != last_db_slug:
                        self.session_store.record_position(st)
                        last_db_rec_t = t0
                        last_db_slug = est.dungeon_slug
                    if self.server:
                        self.server.update_host_position_sync(st)
                    elif self.client:
                        self.client.send_position_sync(st)
                    with self._lock:
                        self._latest_est = est
                except Exception as exc:
                    self.dev_logger.log_event("capture_error", {"error": str(exc)})
                target_end = t0 + interval
                rem = target_end - time.perf_counter()
                if rem > 0.001:
                    self._stop_capture_event.wait(timeout=rem)
        finally:
            cap.close()

    def _poll_tracker(self) -> None:
        if not self.running or self._closing or self.replay_mode:
            return
        with self._lock:
            est = self._latest_est
        if est is None:
            return
        is_new_est = (est is not self._last_rendered_est)
        now_ui = time.perf_counter()
        if (now_ui - getattr(self, "_last_tel_ui_t", 0.0)) >= 0.25:
            self._last_tel_ui_t = now_ui
            tel = self.tracker_mm.get_telemetry()
            cap_ms = getattr(self, "_capture_ms_ema", 0.0)
            backend = getattr(self, "_capture_backend", "")
            ver_part = f" | VerAI: {tel['ver_neural_hz']:.1f}Hz ({tel['ver_neural_ms']:.0f}ms)" if tel["ver_neural_hz"] > 0 else ""
            self.lbl_telemetry.setText(
                f"AI[{tel['provider']}]: {tel['neural_hz']:.1f}Hz ({tel['neural_ms']:.1f}ms){ver_part} | "
                f"CV: {tel['cv_hz']:.1f}Hz ({tel['cv_ms']:.1f}ms) | Cap:{cap_ms:.1f}ms [{backend}•{tel['mode_tag']}]"
            )
            self.pill_fps.setText(f"AI {tel['neural_hz']:.0f} | CV {tel['cv_hz']:.0f} Hz [{tel['provider']}]")

        if is_new_est:
            self._last_rendered_est = est
            self.pill_pos.setText(f"Pos: ({est.world_pos[0]:.0f}, {est.world_pos[1]:.0f}) {est.position_confidence * 100:.0f}%")
            self.lbl_dungeon.setText(f"{est.dungeon_name}  [{est.dungeon_confidence * 100:.0f}% | {len(self.raid_players)} synced]")

        # Time-based 60 FPS visual glide toward the latest tracker position.
        # Adapts the glide time constant to the polling rate (e.g. tau=0.15s at 5 Hz) so
        # both the player marker and follow-camera map shift softly with zero lurch-pause!
        last_vis_t = getattr(self, "_last_vis_t", 0.0)
        vis_dt = max(0.001, min(0.10, now_ui - last_vis_t)) if last_vis_t > 0.0 else 0.016
        self._last_vis_t = now_ui

        target_px = getattr(self.tracker_mm, "last_smooth_px", None) or (float(est.map_px[0]), float(est.map_px[1]))
        prev_slug = getattr(self, "_render_px_slug", None)
        cur_rpx = getattr(self, "_render_px", None)
        moved_vis = False
        snapped = False
        if cur_rpx is None or prev_slug != est.dungeon_slug:
            self._render_px = (float(target_px[0]), float(target_px[1]))
            self._render_px_slug = est.dungeon_slug
            moved_vis = True
            snapped = True
        else:
            dx = float(target_px[0]) - cur_rpx[0]
            dy = float(target_px[1]) - cur_rpx[1]
            dist = math.hypot(dx, dy)
            if dist > 140.0:
                self._render_px = (float(target_px[0]), float(target_px[1]))
                moved_vis = True
                snapped = True
            elif dist > 0.03:
                tau = max(0.045, min(0.24, 0.75 / max(1.0, self.fps)))
                alpha = 1.0 - math.exp(-vis_dt / tau)
                self._render_px = (cur_rpx[0] + dx * alpha, cur_rpx[1] + dy * alpha)
                moved_vis = True

        pan_needs_glide = (
            self.chk_follow.isChecked()
            and self._render_px is not None
            and math.hypot(self._render_px[0] - self.pan_x, self._render_px[1] - self.pan_y) > 0.03
        )
        if self._showing_azeroth_overview and self._transition_state is None:
            if is_new_est:
                self._maybe_autoload_neural_zone()
                if self._is_good_position_fix(est):
                    self._start_fix_transition(est)
        elif self._transition_state is None and (is_new_est or moved_vis or pan_needs_glide):
            if self.chk_follow.isChecked() and self._render_px is not None:
                if snapped:
                    self.pan_x, self.pan_y = self._render_px[0], self._render_px[1]
                else:
                    pdx = self._render_px[0] - self.pan_x
                    pdy = self._render_px[1] - self.pan_y
                    pdist = math.hypot(pdx, pdy)
                    if pdist > 140.0:
                        self.pan_x, self.pan_y = self._render_px[0], self._render_px[1]
                    elif pdist > 0.03:
                        pan_tau = max(0.06, min(0.26, 0.85 / max(1.0, self.fps)))
                        pan_alpha = 1.0 - math.exp(-vis_dt / pan_tau)
                        self.pan_x += pdx * pan_alpha
                        self.pan_y += pdy * pan_alpha
                        moved_vis = True
            if is_new_est:
                self._maybe_autoload_neural_zone()
                self._maybe_stream_nearby_zones(est)
            if self.view_3d_mode:
                if is_new_est or moved_vis:
                    self._render_3d_frame()
            else:
                self.viewport.update()

    def _maybe_autoload_neural_zone(self) -> None:
        """If the ONNX neural model strongly recognizes a WoW: Forever zone not yet loaded, stream it automatically."""
        if self.world_maps is None:
            return
        pred = getattr(self.tracker_mm, "last_neural_pred", None)
        if pred is None or pred.best_prob < 0.55:
            return
        slug = pred.best_slug
        if not slug or slug in self.tracker_3d.dungeons:
            return
        loading_set = getattr(self, "_nearby_loading", None)
        if loading_set is None:
            loading_set = set()
            self._nearby_loading = loading_set
        if slug in loading_set:
            return
        entry = self.world_maps.get_entry(slug)
        if entry is None:
            return
        loading_set.add(slug)

        def _load_nn(s: str = slug) -> None:
            try:
                d = self.world_maps.load_map(s, preview_only=True)
                if not self._closing and d is not None:
                    self._ui_queue.put(("map_cached_register", (s, d)))
            except Exception:
                pass

        threading.Thread(target=_load_nn, daemon=True, name=f"QtNeuralZone-{slug}").start()

    def _maybe_stream_nearby_zones(self, est: TrackerEstimate) -> None:
        """Auto-load any containing or adjacent WoW: Forever city/zone (e.g. Orgrimmar from Durotar)."""
        if self.world_maps is None or est.position_confidence < 0.35:
            return
        now_t = time.perf_counter()
        if now_t - getattr(self, "_last_nearby_check_t", 0.0) < 3.0:
            return
        self._last_nearby_check_t = now_t
        active_entry = self.world_maps.get_entry(est.dungeon_slug)
        if active_entry is None or not active_entry.minimap_folder:
            return
        wx, wy = float(est.world_pos[0]), float(est.world_pos[1])
        folder = active_entry.minimap_folder
        loading_set = getattr(self, "_nearby_loading", None)
        if loading_set is None:
            loading_set = set()
            self._nearby_loading = loading_set
        pad = 320.0
        for e in self._all_map_entries:
            if e.slug in self.tracker_3d.dungeons or e.slug in loading_set:
                continue
            if e.minimap_folder != folder or e.category not in ("Cities", "Zones") or e.bounds_xy is None:
                continue
            xmin, ymin, xmax, ymax = e.bounds_xy
            if (xmin - pad) <= wx <= (xmax + pad) and (ymin - pad) <= wy <= (ymax + pad):
                loading_set.add(e.slug)

                def _load_nb(s: str = e.slug) -> None:
                    try:
                        d = self.world_maps.load_map(s, preview_only=True)
                        if not self._closing and d is not None:
                            self._ui_queue.put(("map_cached_register", (s, d)))
                    except Exception:
                        pass

                threading.Thread(target=_load_nb, daemon=True, name=f"QtNearby-{e.slug}").start()
                break

    # --- UI Queue Drain ---
    def _drain_ui_queue(self) -> None:
        if self._closing:
            return
        needs_update = False
        while True:
            try:
                kind, payload = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "azeroth_overview_loaded":
                ov, rgba = payload
                if ov is not None:
                    if self._azeroth_overview is not None:
                        self._azeroth_overview.close()
                    self._azeroth_overview = ov
                    self._overview_pan_x = ov.width * 0.5
                    self._overview_pan_y = ov.height * 0.5
                    self._overview_zoom = 1.0
                    if rgba is not None:
                        self.viewport.queue_texture_upload_rgba("__azeroth_ov", rgba, build_mips=True)
                    else:
                        self.viewport.queue_texture_upload("__azeroth_ov", np.asarray(ov.image, dtype=np.uint8), build_mips=True)
                    needs_update = True
            elif kind == "cont_l1_loaded":
                cont_slug, res, meta = payload
                self._cont_l1_loading.discard(cont_slug)
                if res is not None:
                    arr_rgba, native_px = res
                    self._cont_l1_native[cont_slug] = native_px
                    if meta is not None:
                        self._cont_meta[cont_slug] = meta
                    self.viewport.queue_texture_upload_rgba(f"__l1_{cont_slug}", arr_rgba, build_mips=True)
                    needs_update = True
            elif kind == "deep_tile_ready":
                req_id, key, arr_rgba = payload
                if getattr(self, "_deep_tile_inflight_key", None) == key:
                    self._deep_tile_inflight_key = None
                if req_id == self._deep_tile_req_id and arr_rgba is not None and not self._is_interacting:
                    self._deep_tile_key = key
                    self.viewport.queue_texture_upload_rgba("__deep_tile", arr_rgba, build_mips=False)
                    needs_update = True
            elif kind == "zone_fullres_loaded":
                slug, arr_rgba = payload
                self._streaming_zones.discard(slug)
                if arr_rgba is not None:
                    self._fullres_uploaded_zones.add(slug)
                    self.viewport.queue_texture_upload_rgba(f"zone_{slug}", arr_rgba, build_mips=True)
                    needs_update = True
            elif kind == "scene_3d_prepared":
                slug, prepared, err = payload
                if getattr(self, "_scene_3d_loading", None) == slug:
                    self._scene_3d_loading = None
                if err is not None or prepared is None:
                    self._scene_3d_error = err or f"No 3D WMO model available for {slug}"
                else:
                    self._scene_3d_error = None
                    self.viewport._pending_3d_scene = (slug, prepared)
                needs_update = True
            elif kind == "map_cached_register":
                slug, dungeon = payload
                if dungeon is not None and slug not in self.tracker_3d.dungeons:
                    self.tracker_3d.register_dungeon(dungeon)
                    self.tracker_mm.register_dungeon(dungeon)
            elif kind == "map_loaded":
                slug, dungeon, err = payload
                if err is not None or dungeon is None:
                    self.lbl_map_status.setText(f"Load failed: {err}")
                else:
                    self.tracker_3d.register_dungeon(dungeon)
                    self.tracker_mm.register_dungeon(dungeon)
                    self.tracker_mm.reset()
                    self.tracker_mm.locked_dungeon_slug = slug
                    self.tracker_mm.active_slug = slug
                    self._latest_est = None
                    self._cancel_fix_transition()
                    self._showing_azeroth_overview = False
                    self.lbl_dungeon.setText(f"{dungeon.name}  [{len(self.raid_players)} synced]")
                    self.lbl_map_status.setText(f"Loaded {dungeon.name}")
                    self._reset_view()
                    needs_update = True
            elif kind == "raid_state":
                self.raid_players = payload
                self._refresh_roster(self.replay_players if self.replay_mode else self.raid_players)
                needs_update = True
            elif kind == "draw_op":
                self.whiteboard.apply_remote_dict(payload)
                needs_update = True
            elif kind == "log":
                self.lbl_stats.setText(str(payload))
            elif kind == "status":
                ok, msg = payload
                self._net_connected = bool(ok)
                if not ok:
                    self.last_net_error = str(msg)
                self.lbl_stats.setText(str(msg))
                self._refresh_sync_pill()
        if needs_update:
            self.viewport.update()

    # --- Networking & Replay ---
    def get_local_lan_ip(self) -> str:
        return str(self.server.local_ip) if self.server and getattr(self.server, "local_ip", None) else get_local_lan_ip()

    def get_network_status_summary(self) -> str:
        if self.server and self.server.is_running:
            return f"Hosting on {self.get_local_lan_ip()}:{self.server.config.port} ({self.server.connected_count} connected)"
        if self.client:
            if self.client.connected:
                return f"Connected to {self.client.url} ({self.client.assigned_role} | offset: {self.client.clock_offset_ms}ms)"
            return f"Connecting to {self.client.url}..."
        return "Offline Solo"

    def _refresh_sync_pill(self) -> None:
        if self.server and self.server.is_running:
            self.pill_sync.setText(f"Host :{self.server.config.port} ({self.server.connected_count})")
            self.pill_sync.setStyleSheet("color: #3fb950; background: #21262d; border: 1px solid #30363d; padding: 3px 8px;")
        elif self.client and self.client.connected:
            self.pill_sync.setText(f"Synced ({self.client.clock_offset_ms}ms)")
            self.pill_sync.setStyleSheet("color: #3fb950; background: #21262d; border: 1px solid #30363d; padding: 3px 8px;")
        else:
            self.pill_sync.setText("Offline Solo")
            self.pill_sync.setStyleSheet("color: #58a6ff; background: #21262d; border: 1px solid #30363d; padding: 3px 8px;")

    def _refresh_roster(self, shown: Dict[str, PlayerState]) -> None:
        sig = tuple((pid, p.name, p.role, p.dungeon_slug, bool(getattr(p, "alive", True))) for pid, p in shown.items())
        if sig == self._last_roster_sig:
            return
        self._last_roster_sig = sig
        self.roster_tree.clear()
        for p in shown.values():
            QTreeWidgetItem(self.roster_tree, [p.name, _role_badge(p.role), p.dungeon_slug or "-", "ALIVE" if p.alive else "DEAD"])
        self.lbl_roster_hdr.setText(f"Guildmates ({len(shown)})")

    def _can_edit_whiteboard(self) -> bool:
        return bool(self.client.can_draw()) if self.client is not None else True

    def _broadcast_draw(self, op_dict: dict) -> None:
        if self.server:
            self.server.broadcast_draw_op_sync(op_dict)
        elif self.client:
            self.client.send_draw_op_sync(op_dict)

    def start_host(self, port: int, passphrase: str, whiteboard_permission: str = "all", officer_token: str = "") -> bool:
        self.stop_network()
        self.last_net_error = None
        self.host_whiteboard_perm = whiteboard_permission
        self.server = ForeverNavigatorServer(
            HostConfig(port=port, room_passphrase=passphrase, officer_token=officer_token, whiteboard_permission=whiteboard_permission),
            on_raid_state=lambda p: self._ui_queue.put(("raid_state", p)),
            on_draw_op=lambda m: self._ui_queue.put(("draw_op", m)),
            on_log=lambda m: self._ui_queue.put(("log", m)),
            on_status=lambda ok, m: self._ui_queue.put(("status", (ok, m))),
        )
        if not self.server.start():
            self.last_net_error = self.server.last_error or f"Failed to bind port {port}"
            self.server = None
            return False
        self._refresh_sync_pill()
        return True

    def start_client(self, url: str, passphrase: str, officer_token: str = "") -> bool:
        self.stop_network()
        self.last_net_error = None
        self.client = ForeverNavigatorClient(
            url=url,
            identity=self.identity,
            prefs=self.sync_prefs,
            room_passphrase=passphrase,
            officer_token=officer_token,
            on_raid_state=lambda p: self._ui_queue.put(("raid_state", p)),
            on_draw_op=lambda m: self._ui_queue.put(("draw_op", m)),
            on_log=lambda m: self._ui_queue.put(("log", m)),
            on_status=lambda ok, m: self._ui_queue.put(("status", (ok, m))),
        )
        if not self.client.start():
            self.last_net_error = self.client.last_error or "Failed to start client"
            self.client = None
            return False
        return True

    def stop_network(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None
        if self.client:
            self.client.stop()
            self.client = None
        self.raid_players.clear()
        self._refresh_sync_pill()
        self._refresh_roster({})

    def _on_replay_toggle(self, enabled: bool) -> None:
        self.replay_mode = bool(enabled)
        if not self.replay_mode:
            self.lbl_scrub.setText("LIVE")
            self.replay_players.clear()
        self.viewport.update()

    def _on_scrub(self, val: int) -> None:
        if not self.replay_mode:
            return
        t0, t1 = self.timeline.bounds_ms()
        if t1 <= t0:
            self.lbl_scrub.setText("No data")
            return
        target_ms = int(t0 + (val / 1000.0) * (t1 - t0))
        snap = self.timeline.scrub_to(target_ms)
        self.replay_players = snap.players
        self.lbl_scrub.setText(f"+{(target_ms - t0) / 1000.0:.1f}s")
        self._refresh_roster(self.replay_players)
        self.viewport.update()

    def closeEvent(self, event) -> None:
        self._closing = True
        self.running = False
        self._stop_capture_event.set()
        for tmr in (self._queue_timer, self._poll_timer, self._zoom_timer, self._settle_timer, self._trans_timer):
            tmr.stop()
        self.stop_network()
        self.viewport.release_gl_resources()
        if self._azeroth_overview is not None:
            self._azeroth_overview.close()
            self._azeroth_overview = None
        self.dev_logger.close()
        self.session_store.close()
        self.tracker_mm.close()
        if self.world_maps is not None and hasattr(self.world_maps, "close"):
            self.world_maps.close()
        self.viewer_3d.close()
        event.accept()


def run_qt_app(index_dir: str | Path = "data/index", fps: Optional[float] = None) -> int:
    """Launch the GPU-accelerated PySide6 + ModernGL ForeverNavigator application."""
    from ..navigator.startup_cleanup import run_startup_cleanup

    run_startup_cleanup(Path(index_dir).parent)

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    # Windows DWM already VSync-composites desktop windows; swapInterval(0) avoids stalling the Qt GUI thread
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = ForeverNavigatorQtWindow(index_dir=index_dir, fps=fps)
    win.show()
    return int(app.exec())
