"""Real GPU-accelerated 3D WMO dungeon viewer (ModernGL) rendering actual WoW 3D models & BLP textures.

Loads the real .wmo geometry, MOCV baked lighting, and .blp textures on demand directly from the
user's local World of Warcraft installation (never redistributing Blizzard 3D assets).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from ..net.protocol import WOW_CLASS_COLORS
from .blp_decoder import decode_blp
from .casc_reader import LocalCASC, load_listfile
from .spawns import get_spawns_for_dungeon
from .tracker import DungeonData
from .wmo_parser import WMOGroup, WMORoot, parse_wmo_group, parse_wmo_root, resolve_material_texture

# Root WMO FileDataIDs for WoW: Forever dungeons & raids
DUNGEON_WMO_FDIDS: Dict[str, int] = {
    "ragefire_chasm": 109483,
    "ruins_of_lordaeron": 114612,
    "stockade": 108658,
    "deadmines": 108444,
    "shadowfang_keep": 109971,
    "blackfathom_deeps": 109179,
    "blackrock_depths": 108239,
    "blackrock_spire": 108348,
    "molten_core": 108286,
    "dire_maul": 109280,
    "gnomeregan": 5568954,
    "maraudon": 109415,
    "onyxias_lair": 109474,
    "sunken_temple": 5742583,
    "uldaman": 109782,
    "wailing_caverns": 109550,
}

VERT_SHADER = """
#version 330 core
in vec3 in_pos;
in vec3 in_norm;
in vec2 in_uv;
in vec4 in_col;

uniform mat4 u_mvp;

out vec3 v_norm;
out vec2 v_uv;
out vec4 v_col;
out vec3 v_pos;

void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_norm = in_norm;
    v_uv = in_uv;
    v_col = in_col;
    v_pos = in_pos;
}
"""

FRAG_SHADER = """
#version 330 core
in vec3 v_norm;
in vec2 v_uv;
in vec4 v_col;
in vec3 v_pos;

uniform sampler2D u_tex;
uniform int u_cutaway;
uniform float u_brightness;

out vec4 fragColor;

void main() {
    // Cutaway mode: hide downward-facing ceilings so you can see cleanly inside the 3D dungeon
    if (u_cutaway == 1 && v_norm.z < -0.35) {
        discard;
    }
    vec4 tex = texture(u_tex, v_uv);
    if (tex.a < 0.25) {
        discard;
    }
    vec3 n = normalize(v_norm);
    float ndl = 0.58 + 0.42 * max(dot(n, normalize(vec3(0.35, 0.25, 0.90))), 0.0);
    vec3 baked = mix(vec3(0.78), v_col.rgb * 1.55, 0.72);
    vec3 col = tex.rgb * baked * ndl * u_brightness;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def _hex_to_rgb(hex_color: str, fallback: Tuple[int, int, int] = (255, 90, 70)) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' hex string to an (R, G, B) tuple for OpenCV drawing on RGB canvases."""
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return fallback
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return fallback


def _role_badge(role: str) -> str:
    r = (role or "DPS").upper()
    if r == "LEADER":
        return "L"
    if r == "OFFICER":
        return "O"
    if r == "TANK":
        return "T"
    if r == "HEALER":
        return "H"
    return "D"


def find_wow_dir(preferred: Optional[str | Path] = None) -> Optional[Path]:
    """Auto-detect local World of Warcraft installation folder (accepts root or _forever_ subfolder)."""
    candidates = []
    if preferred:
        p = Path(preferred)
        candidates.extend([p, p.parent])
    candidates.extend([
        Path(r"C:\Program Files (x86)\World of Warcraft"),
        Path(r"C:\Program Files\World of Warcraft"),
        Path(r"D:\World of Warcraft"),
        Path(r"D:\Games\World of Warcraft"),
        Path(r"E:\World of Warcraft"),
        Path(r"E:\Spel\World of Warcraft"),
    ])
    for p in candidates:
        if (p / ".build.info").exists() and (p / "Data").exists():
            return p
    return None


class _LoadedGPUScene:
    """Holds GPU buffers & textures for one loaded WMO dungeon."""

    def __init__(
        self,
        ctx: moderngl.Context,
        prog: moderngl.Program,
        default_tex: moderngl.Texture,
        root: WMORoot,
        groups: List[WMOGroup],
        casc: LocalCASC,
        path_to_id: Dict[str, int],
    ):
        self.draw_calls: List[Tuple[moderngl.VertexArray, moderngl.Texture]] = []
        self._buffers: List[moderngl.Buffer] = []
        self._textures: List[moderngl.Texture] = []

        # Compute bounding box & center
        all_v = np.concatenate([g.vertices for g in groups if len(g.vertices) > 0], axis=0)
        self.bbox_min = all_v.min(axis=0).astype(np.float32)
        self.bbox_max = all_v.max(axis=0).astype(np.float32)
        self.center = 0.5 * (self.bbox_min + self.bbox_max)
        self.radius = float(np.linalg.norm(self.bbox_max - self.bbox_min) * 0.5)

        # Decode & upload real Blizzard BLP2 textures
        tex_by_fdid: Dict[int, moderngl.Texture] = {}
        mat_textures: Dict[int, moderngl.Texture] = {}
        for mat_idx, mat in enumerate(root.materials):
            fdid = resolve_material_texture(root, mat, path_to_id)
            if fdid is None:
                continue
            if fdid in tex_by_fdid:
                mat_textures[mat_idx] = tex_by_fdid[fdid]
                continue
            try:
                img = decode_blp(casc.read_file(fdid)).convert("RGBA")
                if img.width > 256 or img.height > 256:
                    img = img.resize((min(256, img.width), min(256, img.height)), Image.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                tex = ctx.texture((arr.shape[1], arr.shape[0]), 4, arr.tobytes())
                tex.repeat_x = True
                tex.repeat_y = True
                tex.build_mipmaps()
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tex_by_fdid[fdid] = tex
                mat_textures[mat_idx] = tex
                self._textures.append(tex)
            except Exception:
                continue

        # Upload real WMO group triangle geometry to GPU VBOs/IBOs
        for g in groups:
            if len(g.vertices) == 0 or len(g.indices) < 3:
                continue
            interleaved = np.hstack([g.vertices, g.normals, g.uvs, g.vertex_colors]).astype(np.float32)
            vbo = ctx.buffer(interleaved.tobytes())
            self._buffers.append(vbo)
            for b in g.batches:
                sub_idx = g.indices[b.start_index : b.start_index + b.count]
                if len(sub_idx) < 3:
                    continue
                ibo = ctx.buffer(sub_idx.astype(np.uint32).tobytes())
                self._buffers.append(ibo)
                vao = ctx.vertex_array(
                    prog,
                    [(vbo, "3f 3f 2f 4f", "in_pos", "in_norm", "in_uv", "in_col")],
                    index_buffer=ibo,
                )
                tex = mat_textures.get(b.material_id, default_tex)
                self.draw_calls.append((vao, tex))

    def release(self) -> None:
        for vao, _ in self.draw_calls:
            vao.release()
        for b in self._buffers:
            b.release()
        for t in self._textures:
            t.release()


class GPU3DViewer:
    """Real GPU-rendered 3D WMO dungeon viewer using ModernGL + local WoW CASC files."""

    def __init__(
        self,
        wow_dir: Optional[str | Path] = None,
        listfile_path: Optional[str | Path] = None,
        product: str = "wowf",
    ):
        self.wow_dir = find_wow_dir(wow_dir)
        self.listfile_path = Path(listfile_path) if listfile_path else None
        self.product = product

        # Lazy-initialized OpenGL context & CASC reader (zero overhead until user toggles 3D!)
        self.ctx: Optional[moderngl.Context] = None
        self.prog: Optional[moderngl.Program] = None
        self.default_tex: Optional[moderngl.Texture] = None
        self._fbo: Optional[moderngl.Framebuffer] = None
        self._fbo_color: Optional[moderngl.Texture] = None
        self._fbo_depth: Optional[moderngl.Renderbuffer] = None
        self._fbo_size: Tuple[int, int] = (0, 0)

        self._casc: Optional[LocalCASC] = None
        self._id_to_path: Dict[int, str] = {}
        self._path_to_id: Dict[str, int] = {}

        self._loaded_slug: Optional[str] = None
        self._scene: Optional[_LoadedGPUScene] = None
        self._error_msg: Optional[str] = None

        # Orbit camera state
        self.yaw_deg: float = -45.0
        self.pitch_deg: float = 48.0
        self.zoom: float = 1.0
        self.cutaway_ceilings: bool = True
        self.brightness: float = 1.0
        self.target: Optional[np.ndarray] = None
        self._last_player_z: Optional[float] = None

    def set_wow_dir(self, wow_dir: str | Path) -> None:
        """Switch WoW installation path and reload CASC cleanly on next 3D render."""
        new_dir = find_wow_dir(wow_dir)
        if new_dir == self.wow_dir:
            return
        if self._scene is not None:
            self._scene.release()
            self._scene = None
        if self._casc is not None:
            self._casc.close()
            self._casc = None
        self._loaded_slug = None
        self._error_msg = None
        self.wow_dir = new_dir

    def _ensure_gl(self) -> None:
        if self.ctx is not None:
            return
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)
        base = np.full((32, 32, 4), 165, dtype=np.uint8)
        base[:, :, 3] = 255
        self.default_tex = self.ctx.texture((32, 32), 4, base.tobytes())

    def _ensure_fbo(self, width: int, height: int) -> None:
        if self._fbo is not None and self._fbo_size == (width, height):
            return
        if self._fbo is not None:
            self._fbo.release()
            self._fbo_color.release()
            self._fbo_depth.release()
        assert self.ctx is not None
        self._fbo_color = self.ctx.texture((width, height), 3)
        self._fbo_depth = self.ctx.depth_renderbuffer((width, height))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._fbo_color], depth_attachment=self._fbo_depth
        )
        self._fbo_size = (width, height)

    def _ensure_casc(self) -> bool:
        if self._casc is not None:
            return True
        if self.wow_dir is None:
            self._error_msg = "Local WoW folder not found (needed for real 3D WMO models)"
            return False
        try:
            self._casc = LocalCASC(self.wow_dir, product=self.product)
            self._casc._ensure_root()
            if self.listfile_path and self.listfile_path.exists():
                self._id_to_path, self._path_to_id = load_listfile(self.listfile_path)
            return True
        except Exception as exc:
            self._error_msg = f"Failed to open WoW CASC: {exc}"
            return False

    def _ensure_scene(self, slug: str) -> Optional[_LoadedGPUScene]:
        if slug == self._loaded_slug:
            return self._scene
        # Release previous dungeon's GPU memory so VRAM stays tiny
        if self._scene is not None:
            self._scene.release()
            self._scene = None
        self._loaded_slug = slug
        self._error_msg = None

        fdid = DUNGEON_WMO_FDIDS.get(slug)
        if fdid is None:
            self._error_msg = "Outdoor BLP map (no single WMO 3D model) — use 2D Map view"
            return None

        self._ensure_gl()
        if not self._ensure_casc():
            return None
        assert self._casc is not None and self.ctx is not None and self.prog is not None and self.default_tex is not None

        if not self._casc.has_file(fdid):
            self._error_msg = f"WMO model FDID {fdid} not found in local {self.product} CASC"
            return None

        root = parse_wmo_root(self._casc.read_file(fdid))
        group_fdids = list(root.group_fdids)
        if not group_fdids and fdid in self._id_to_path:
            stem = self._id_to_path[fdid][:-4]
            for gi in range(root.n_groups):
                gp = f"{stem}_{gi:03d}.wmo".lower()
                if gp in self._path_to_id:
                    group_fdids.append(self._path_to_id[gp])

        groups: List[WMOGroup] = []
        for gi, gfdid in enumerate(group_fdids[:160]):
            if gfdid == 0 or not self._casc.has_file(gfdid):
                continue
            try:
                grp = parse_wmo_group(self._casc.read_file(gfdid), group_index=gi)
                if grp is not None:
                    groups.append(grp)
            except Exception:
                continue

        if not groups:
            self._error_msg = "No 3D WMO groups could be loaded"
            return None

        self._scene = _LoadedGPUScene(
            self.ctx, self.prog, self.default_tex, root, groups, self._casc, self._path_to_id
        )
        self.target = self._scene.center.copy()
        return self._scene

    def reset_camera(self, slug: str) -> None:
        self.yaw_deg = -45.0
        self.pitch_deg = 48.0
        self.zoom = 1.0
        if self._scene is not None and self._loaded_slug == slug:
            self.target = self._scene.center.copy()

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw_deg = (self.yaw_deg - dx * 0.45) % 360.0
        self.pitch_deg = float(np.clip(self.pitch_deg + dy * 0.35, 8.0, 86.0))

    def pan(self, dx: float, dy: float, _slug: str) -> None:
        if self._scene is None:
            return
        if self.target is None:
            self.target = self._scene.center.copy()
        yaw = math.radians(self.yaw_deg)
        right = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
        fwd = np.array([-math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float32)
        speed = (self._scene.radius * 0.0028) / max(0.3, self.zoom)
        self.target = self.target - right * (dx * speed) + fwd * (dy * speed)

    def snap_world_z(self, d: DungeonData, world_xy: np.ndarray, prefer_z: Optional[float] = None) -> float:
        """Snap a 2D (x, y) estimate to the nearest walkable 3D floor waypoint, preferring current floor Z."""
        if len(d.waypoints) == 0:
            return float(self._scene.center[2]) if self._scene is not None else 0.0
        dxy2 = np.sum((d.waypoints[:, :2] - world_xy[None, :2]) ** 2, axis=1)
        if prefer_z is not None:
            # Penalize sudden vertical floor jumps unless XY clearly moved to another ramp/floor
            dz2 = (d.waypoints[:, 2] - float(prefer_z)) ** 2
            score = dxy2 + 0.45 * dz2
            return float(d.waypoints[int(np.argmin(score)), 2])
        return float(d.waypoints[int(np.argmin(dxy2)), 2])

    def render(
        self,
        slug: str,
        d: DungeonData,
        width: int,
        height: int,
        player_world: Optional[np.ndarray] = None,
        trail_world: Optional[List[np.ndarray]] = None,
        follow_player: bool = True,
        show_spawns: bool = True,
        raid_players: Optional[Sequence[Any]] = None,
        local_player_id: Optional[str] = None,
    ) -> np.ndarray:
        """Render the actual textured 3D WMO dungeon on the GPU via ModernGL."""
        width = max(1, int(width))
        height = max(1, int(height))
        scene = self._ensure_scene(slug)
        if scene is None:
            canvas = np.full((height, width, 3), (18, 21, 28), dtype=np.uint8)
            msg = self._error_msg or f"No 3D WMO model available for {d.name}"
            cv2.putText(canvas, msg, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 195, 115), 1, cv2.LINE_AA)
            return canvas

        if self.target is None:
            self.target = scene.center.copy()

        p3d: Optional[np.ndarray] = None
        if player_world is not None:
            pz = self.snap_world_z(d, player_world[:2], prefer_z=self._last_player_z)
            self._last_player_z = pz
            p3d = np.array([float(player_world[0]), float(player_world[1]), pz], dtype=np.float32)
            if follow_player:
                self.target = p3d.copy()

        # Orbit camera
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        dist = max(18.0, (scene.radius * 1.95) / max(0.30, self.zoom))
        cam_dir = np.array(
            [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)],
            dtype=np.float32,
        )
        eye = self.target + cam_dir * dist
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        view = Matrix44.look_at(eye, self.target, up, dtype="f4")
        proj = Matrix44.perspective_projection(55.0, width / float(max(1, height)), 0.5, 2500.0, dtype="f4")
        mvp = proj * view

        self._ensure_fbo(width, height)
        assert self.ctx is not None and self._fbo is not None and self.prog is not None

        self._fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.07, 0.08, 0.11, 1.0)
        self.prog["u_mvp"].write(mvp.astype("f4").tobytes())
        self.prog["u_cutaway"].value = 1 if self.cutaway_ceilings else 0
        self.prog["u_brightness"].value = float(self.brightness)
        self.prog["u_tex"].value = 0

        for vao, tex in scene.draw_calls:
            tex.use(location=0)
            vao.render()

        raw = self._fbo.read(components=3)
        canvas = np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)).copy()

        # Helper to project 3D world points through the exact same OpenGL MVP matrix
        mvp_mat = np.array(mvp, dtype=np.float32)  # pyrr row-major: v_clip = [x,y,z,1] @ mvp_mat

        def project_3d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            ones = np.ones((len(pts), 1), dtype=np.float32)
            clip = np.hstack([pts, ones]) @ mvp_mat
            w_c = clip[:, 3]
            valid = w_c > 0.5
            ndc_x = np.where(valid, clip[:, 0] / np.maximum(0.5, w_c), -99.0)
            ndc_y = np.where(valid, clip[:, 1] / np.maximum(0.5, w_c), -99.0)
            sx = (ndc_x * 0.5 + 0.5) * width
            sy = (0.5 - ndc_y * 0.5) * height
            return sx, sy, valid

        # 3D movement trail
        if trail_world and len(trail_world) >= 2:
            t_pts = np.array(
                [[float(p[0]), float(p[1]), self.snap_world_z(d, p[:2]) + 2.0] for p in trail_world],
                dtype=np.float32,
            )
            tu, tv, tv_ok = project_3d(t_pts)
            pts_2d = [(int(round(tu[i])), int(round(tv[i]))) for i in range(len(t_pts)) if tv_ok[i]]
            if len(pts_2d) >= 2:
                cv2.polylines(canvas, [np.array(pts_2d, dtype=np.int32)], False, (100, 235, 175), 2, cv2.LINE_AA)

        # Floating 3D boss & entrance spawn pins
        if show_spawns:
            spawns = get_spawns_for_dungeon(slug)
            if spawns:
                map_h = d.map_image.shape[0]
                sp_pts = []
                for sp in spawns:
                    wx, wy = d.map_px_to_world(sp.u * map_h, sp.v * map_h)
                    wz = self.snap_world_z(d, np.array([wx, wy], dtype=np.float32))
                    sp_pts.append([wx, wy, wz])
                sp_arr = np.array(sp_pts, dtype=np.float32)
                sp_top = sp_arr.copy()
                sp_top[:, 2] += 14.0
                bu, bv, bok = project_3d(sp_arr)
                pu, pv, pok = project_3d(sp_top)
                for i, sp in enumerate(spawns):
                    if not (bok[i] and pok[i]):
                        continue
                    bx, by = int(round(bu[i])), int(round(bv[i]))
                    px, py = int(round(pu[i])), int(round(pv[i]))
                    if not (0 <= px < width and 0 <= py < height):
                        continue
                    pin_col = (90, 220, 255) if sp.kind == "entrance" else ((255, 165, 60) if sp.kind == "boss" else (180, 195, 220))
                    cv2.line(canvas, (bx, by), (px, py), pin_col, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 5, pin_col, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
                    (tw, th), _ = cv2.getTextSize(sp.label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                    lx, ly = px + 8, py + 4
                    cv2.rectangle(canvas, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (14, 17, 24), -1)
                    cv2.putText(canvas, sp.label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.44, pin_col, 1, cv2.LINE_AA)

        # Connected Guildmates / Raid Players 3D Beacons
        rendered_local_in_raid = False
        if raid_players:
            for p in raid_players:
                if getattr(p, "dungeon_slug", "") != slug:
                    continue
                pid = getattr(p, "player_id", "")
                if local_player_id and pid == local_player_id:
                    rendered_local_in_raid = True
                px_w = float(getattr(p, "x", 0.0))
                py_w = float(getattr(p, "y", 0.0))
                pz_raw = float(getattr(p, "z", 0.0))
                pz = self.snap_world_z(
                    d,
                    np.array([px_w, py_w], dtype=np.float32),
                    prefer_z=pz_raw if abs(pz_raw) > 1e-3 else None,
                )
                base_pt = np.array([[px_w, py_w, pz]], dtype=np.float32)
                top_pt = np.array([[px_w, py_w, pz + 16.0]], dtype=np.float32)
                bu, bv, bok = project_3d(base_pt)
                tu, tv, tok = project_3d(top_pt)
                if not (bok[0] and tok[0]):
                    continue
                bx, by = int(round(bu[0])), int(round(bv[0]))
                tx, ty = int(round(tu[0])), int(round(tv[0]))
                if not (-40 <= tx < width + 40 and -40 <= ty < height + 40):
                    continue

                alive = bool(getattr(p, "alive", True))
                wow_class = str(getattr(p, "wow_class", "UNKNOWN"))
                col = _hex_to_rgb(WOW_CLASS_COLORS.get(wow_class, "#ff5a46")) if alive else (135, 140, 150)
                outline_col = (245, 201, 107) if (local_player_id and pid == local_player_id) else (255, 255, 255)

                cv2.line(canvas, (bx, by), (tx, ty), col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (bx, by), 7, col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 6, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 6, outline_col, 2 if pid == local_player_id else 1, cv2.LINE_AA)
                if not alive:
                    cv2.line(canvas, (tx - 5, ty - 5), (tx + 5, ty + 5), (229, 62, 62), 2, cv2.LINE_AA)
                    cv2.line(canvas, (tx - 5, ty + 5), (tx + 5, ty - 5), (229, 62, 62), 2, cv2.LINE_AA)

                badge = _role_badge(str(getattr(p, "role", "DPS")))
                name = str(getattr(p, "name", "Raider"))
                label = f"[{badge}|DEAD] {name}" if not alive else f"[{badge}] {name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                lx, ly = tx + 9, ty + 4
                cv2.rectangle(canvas, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (14, 17, 24), -1)
                txt_col = (252, 129, 129) if not alive else col
                cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_col, 1, cv2.LINE_AA)

        # Glowing 3D Player Beacon (fallback when local player isn't in raid_players)
        if p3d is not None and not rendered_local_in_raid:
            bu, bv, bok = project_3d(p3d[None, :])
            tu, tv, tok = project_3d((p3d + np.array([0.0, 0.0, 18.0], dtype=np.float32))[None, :])
            if bok[0] and tok[0]:
                bx, by = int(round(bu[0])), int(round(bv[0]))
                tx, ty = int(round(tu[0])), int(round(tv[0]))
                cv2.line(canvas, (bx, by), (tx, ty), (255, 100, 75), 3, cv2.LINE_AA)
                cv2.circle(canvas, (bx, by), 9, (255, 90, 70), 2, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 7, (255, 80, 65), -1, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"GPU 3D Model (WMO) | Left-drag: Orbit | Right-drag: Pan | Scroll: Zoom ({int(self.zoom*100)}%)",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 210, 225),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def close(self) -> None:
        if self._scene is not None:
            self._scene.release()
            self._scene = None
        if self._fbo is not None:
            self._fbo.release()
            self._fbo_color.release()
            self._fbo_depth.release()
            self._fbo = None
        if self.default_tex is not None:
            self.default_tex.release()
        if self.prog is not None:
            self.prog.release()
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None
        if self._casc is not None:
            self._casc.close()
            self._casc = None
