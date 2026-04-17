"""斑馬線 ROI 多邊形:point-in-polygon、邊界距離。

以影像像素座標表示。世界座標的距離由 violation_rules 配合 homography 計算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from shapely.geometry import Point, Polygon


@dataclass(slots=True)
class RoiPolygon:
    """影像座標下的斑馬線多邊形。"""

    points: list[tuple[float, float]]
    _shapely: Polygon = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"ROI 至少需要 3 個頂點,收到 {len(self.points)}")
        self._shapely = Polygon(self.points)
        if not self._shapely.is_valid:
            raise ValueError("ROI 多邊形無效(例如自相交)")

    def contains_point(self, x: float, y: float) -> bool:
        """點是否在多邊形內(含邊界)。"""
        return bool(self._shapely.covers(Point(x, y)))

    def distance_to_point(self, x: float, y: float) -> float:
        """像素距離。世界距離請透過 homography 換算後的 world_roi 計算。"""
        return float(self._shapely.distance(Point(x, y)))

    def to_json(self) -> dict:
        return {"polygon": [list(p) for p in self.points]}

    @classmethod
    def from_json(cls, data: dict) -> Self:
        pts = data.get("polygon", [])
        return cls(points=[(float(p[0]), float(p[1])) for p in pts])

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
