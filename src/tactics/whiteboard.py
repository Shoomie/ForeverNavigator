"""Collaborative tactical whiteboard (freehand strokes, range circles, text annotations)."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WhiteboardOp:
    op_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    author_id: str = "local"
    dungeon_slug: str = ""
    kind: str = "stroke"  # "stroke", "circle_range", "text"
    color: str = "#ffe066"
    width: int = 3
    points: List[Tuple[float, float]] = field(default_factory=list)  # normalized [0..1] map coords
    text: str = ""
    radius_yd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [list(p) for p in self.points]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WhiteboardOp":
        raw_pts = d.get("points", [])
        pts = [
            (float(p[0]), float(p[1]))
            for p in raw_pts
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        return cls(
            op_id=str(d.get("op_id") or uuid.uuid4().hex[:10]),
            author_id=str(d.get("author_id", "local")),
            dungeon_slug=str(d.get("dungeon_slug", "")),
            kind=str(d.get("kind", "stroke")),
            color=str(d.get("color", "#ffe066")),
            width=int(d.get("width", 3)),
            points=pts,
            text=str(d.get("text", "")),
            radius_yd=float(d.get("radius_yd", 0.0)),
        )


class TacticalWhiteboard:
    """Stores per-dungeon collaborative annotations with undo/clear and network sync."""

    def __init__(self) -> None:
        self._ops_by_dungeon: Dict[str, List[WhiteboardOp]] = {}

    def ops_for(self, dungeon_slug: str) -> List[WhiteboardOp]:
        return self._ops_by_dungeon.get(dungeon_slug, [])

    def add_op(self, op: WhiteboardOp) -> None:
        bucket = self._ops_by_dungeon.setdefault(op.dungeon_slug, [])
        if any(existing.op_id == op.op_id for existing in bucket):
            return
        bucket.append(op)

    def undo(self, dungeon_slug: str) -> Optional[WhiteboardOp]:
        bucket = self._ops_by_dungeon.get(dungeon_slug)
        if bucket:
            return bucket.pop()
        return None

    def clear(self, dungeon_slug: str) -> None:
        self._ops_by_dungeon[dungeon_slug] = []

    def apply_remote_dict(self, msg: Dict[str, Any]) -> None:
        kind = msg.get("kind")
        slug = str(msg.get("dungeon_slug", ""))
        if kind == "clear_all":
            self.clear(slug)
        elif kind == "undo":
            self.undo(slug)
        else:
            self.add_op(WhiteboardOp.from_dict(msg))
