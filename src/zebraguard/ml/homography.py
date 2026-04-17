"""透視變換:影像像素 ↔ 世界座標(公尺)。

使用者在校正 UI 選取 ≥ 4 組地面對應點並輸入實際距離,
由 `from_points` 建構轉換矩陣。後續所有「實際距離」都經由此矩陣換算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np


@dataclass(slots=True)
class Homography:
    """影像 → 世界(公尺)的 3×3 單應性矩陣。"""

    matrix: np.ndarray
    image_points: list[tuple[float, float]]
    world_points_meters: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Homography 需 3x3 矩陣,收到 {self.matrix.shape}")

    @classmethod
    def from_points(
        cls,
        image_points: list[tuple[float, float]],
        world_points_meters: list[tuple[float, float]],
    ) -> Self:
        if len(image_points) != len(world_points_meters):
            raise ValueError("image_points 與 world_points 數量需相等")
        if len(image_points) < 4:
            raise ValueError(f"至少需要 4 組對應點,收到 {len(image_points)}")

        src = np.asarray(image_points, dtype=np.float64)
        dst = np.asarray(world_points_meters, dtype=np.float64)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise RuntimeError("findHomography 失敗;請檢查對應點是否共線或退化")
        return cls(
            matrix=matrix,
            image_points=[(float(p[0]), float(p[1])) for p in image_points],
            world_points_meters=[(float(p[0]), float(p[1])) for p in world_points_meters],
        )

    def image_to_world(self, x: float, y: float) -> tuple[float, float]:
        pts = np.array([[[x, y]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pts, self.matrix)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def image_to_world_many(self, points: np.ndarray) -> np.ndarray:
        """批次轉換。輸入 shape (N, 2),輸出 shape (N, 2)。"""
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def world_distance(
        self, img_p1: tuple[float, float], img_p2: tuple[float, float]
    ) -> float:
        """兩個影像點在世界平面上的直線距離(公尺)。"""
        w1 = np.asarray(self.image_to_world(*img_p1))
        w2 = np.asarray(self.image_to_world(*img_p2))
        return float(np.linalg.norm(w1 - w2))

    def to_json(self) -> dict:
        return {
            "image_points": [list(p) for p in self.image_points],
            "world_points_meters": [list(p) for p in self.world_points_meters],
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> Self:
        if "matrix" in data and data["matrix"]:
            return cls(
                matrix=np.asarray(data["matrix"], dtype=np.float64),
                image_points=[(float(p[0]), float(p[1])) for p in data.get("image_points", [])],
                world_points_meters=[
                    (float(p[0]), float(p[1])) for p in data.get("world_points_meters", [])
                ],
            )
        return cls.from_points(
            [(float(p[0]), float(p[1])) for p in data["image_points"]],
            [(float(p[0]), float(p[1])) for p in data["world_points_meters"]],
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
