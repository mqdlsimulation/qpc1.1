# modules/gate_structures/trenchgate.py

"""
trenchgate.py

Trench gate geometry generator.

Trench gate는 split gate와 수직으로 배치되는 직사각형 게이트입니다.
- Split gate: x축 방향으로 constriction 형성
- Trench gate: x축 방향으로 길이, y축 방향으로 폭을 가짐

좌표계:
- x축: trench gate 길이 방향 (split gate gap을 가로지름)
- y축: trench gate 폭 방향
- z = 0: 웨이퍼 표면 (trench gate가 형성되는 위치)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

Point = Tuple[float, float]


# ---------------- Trench gate 파라미터 ---------------- #

@dataclass
class TrenchGateConfig:
    """
    Trench gate 구조를 정의하는 파라미터.

    x_length     : x 방향 길이 (nm), split gate gap을 가로지르는 방향
    y_width      : y 방향 폭 (nm)
    x_offset     : x 방향 중심 오프셋 (nm), 기본값 0
    y_offset     : y 방향 중심 오프셋 (nm), 기본값 0
    two_deg_depth: 2DEG 깊이 (nm)
    """
    x_length: float
    y_width: float
    x_offset: float = 0.0
    y_offset: float = 0.0
    two_deg_depth: Optional[float] = None


# ---------------- Geometry 생성 함수 ---------------- #

def _rectangle_vertices_centered(
    center: Point,
    x_length: float,
    y_width: float,
) -> List[Point]:
    """
    중심 좌표와 x길이/y폭으로 직사각형 꼭짓점 생성.
    """
    cx, cy = center
    hx = x_length / 2.0
    hy = y_width / 2.0

    return [
        (cx - hx, cy - hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
        (cx - hx, cy + hy),
    ]


def make_trench_gate_shape(
    x_length: float,
    y_width: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Dict[str, List[Point]]:
    """
    Trench gate geometry 생성.

    파라미터:
        x_length : x 방향 길이 (nm)
        y_width  : y 방향 폭 (nm)
        x_offset : x 방향 중심 오프셋 (nm)
        y_offset : y 방향 중심 오프셋 (nm)

    반환:
        {"trench": [(x1, y1), ..., (x4, y4)]}
    """
    center = (x_offset, y_offset)
    vertices = _rectangle_vertices_centered(center, x_length, y_width)

    return {"trench": vertices}


# ---------------- 교차 검사 함수 ---------------- #

def _polygons_overlap(poly1: List[Point], poly2: List[Point]) -> bool:
    """
    두 볼록 다각형(convex polygon)의 교차 여부를 검사.
    Separating Axis Theorem (SAT) 사용.

    반환:
        True if 교차, False if 교차하지 않음
    """
    def get_edges(polygon: List[Point]) -> List[Tuple[float, float]]:
        """다각형의 모든 edge 벡터를 반환."""
        edges = []
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            edges.append(edge)
        return edges

    def get_perpendicular(edge: Tuple[float, float]) -> Tuple[float, float]:
        """edge에 수직인 축 반환."""
        return (-edge[1], edge[0])

    def project_polygon(polygon: List[Point], axis: Tuple[float, float]) -> Tuple[float, float]:
        """다각형을 축에 투영하여 (min, max) 반환."""
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in polygon]
        return (min(dots), max(dots))

    def projections_overlap(proj1: Tuple[float, float], proj2: Tuple[float, float]) -> bool:
        """두 투영 구간이 겹치는지 확인."""
        return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

    # 두 다각형의 모든 edge에서 분리축 후보 생성
    edges = get_edges(poly1) + get_edges(poly2)

    for edge in edges:
        axis = get_perpendicular(edge)

        # 축의 길이가 0이면 스킵
        length = np.sqrt(axis[0]**2 + axis[1]**2)
        if length < 1e-10:
            continue

        # 정규화
        axis = (axis[0] / length, axis[1] / length)

        proj1 = project_polygon(poly1, axis)
        proj2 = project_polygon(poly2, axis)

        if not projections_overlap(proj1, proj2):
            # 분리축 발견 → 교차하지 않음
            return False

    # 모든 축에서 겹침 → 교차함
    return True


def check_trench_splitgate_overlap(
    trench_shapes: Dict[str, List[Point]],
    split_shapes: Dict[str, List[Point]],
) -> None:
    """
    Trench gate와 split gate의 교차 여부를 검사.
    교차 시 ValueError 발생.

    파라미터:
        trench_shapes: {"trench": [...]} 형태
        split_shapes : {"left": [...], "right": [...]} 형태
    """
    trench_poly = trench_shapes.get("trench")
    if trench_poly is None:
        raise KeyError("trench_shapes must contain 'trench' key.")

    for name, split_poly in split_shapes.items():
        if _polygons_overlap(trench_poly, split_poly):
            raise ValueError(
                f"Trench gate overlaps with split gate '{name}'. "
                f"Please adjust gate dimensions or positions."
            )


# ---------------- TrenchGateStructure 클래스 ---------------- #

class TrenchGateStructure:
    """
    Trench gate geometry를 생성하고 관리하는 클래스.
    """

    def __init__(self, config: TrenchGateConfig) -> None:
        self.config = config
        self.shapes: Optional[Dict[str, List[Point]]] = None

    def build_shapes(self) -> None:
        """Trench gate geometry 생성."""
        self.shapes = make_trench_gate_shape(
            x_length=self.config.x_length,
            y_width=self.config.y_width,
            x_offset=self.config.x_offset,
            y_offset=self.config.y_offset,
        )

    def check_overlap_with_splitgate(
        self,
        split_shapes: Dict[str, List[Point]],
    ) -> None:
        """
        Split gate와의 교차 검사.
        교차 시 ValueError 발생.
        """
        if self.shapes is None:
            raise RuntimeError("Shapes have not been built yet. Call build_shapes() first.")

        check_trench_splitgate_overlap(self.shapes, split_shapes)

    def describe(self) -> None:
        """현재 geometry의 꼭짓점 좌표를 텍스트로 출력."""
        if self.shapes is None:
            raise RuntimeError("Shapes have not been built yet. Call build_shapes() first.")

        print("[INFO] Trench gate vertex coordinates (x, y) in nm:")
        for name, verts in self.shapes.items():
            print(f"  {name}:")
            for (x, y) in verts:
                print(f"    ({x:8.3f}, {y:8.3f})")
            print()

    def plot(self, split_shapes: Optional[Dict[str, List[Point]]] = None) -> None:
        """
        Trench gate geometry 시각화 (top view).
        split_shapes가 주어지면 함께 표시.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Trench gate 플롯
        if self.shapes is not None:
            for name, verts in self.shapes.items():
                xs = [v[0] for v in verts] + [verts[0][0]]
                ys = [v[1] for v in verts] + [verts[0][1]]
                ax.fill(xs, ys, alpha=0.3, label=f"{name} gate")
                ax.plot(xs, ys, linewidth=2)

        # Split gate 플롯 (옵션)
        if split_shapes is not None:
            for name, verts in split_shapes.items():
                xs = [v[0] for v in verts] + [verts[0][0]]
                ys = [v[1] for v in verts] + [verts[0][1]]
                ax.fill(xs, ys, alpha=0.3, label=f"split {name}")
                ax.plot(xs, ys, linewidth=2)

        ax.axvline(0, color="k", linestyle="--", alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title("Gate geometry - Top view (x-y)")
        ax.legend(loc="best")

        plt.show()


# ---------------- Helper 함수 ---------------- #

def build_trench_gate_structure(
    x_length: float,
    y_width: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    two_deg_depth: Optional[float] = None,
    split_shapes: Optional[Dict[str, List[Point]]] = None,
    do_describe: bool = True,
    do_plot: bool = False,
    do_overlap_check: bool = True,
) -> TrenchGateStructure:
    """
    Trench gate structure를 생성하는 helper 함수.

    파라미터:
        x_length       : x 방향 길이 (nm)
        y_width        : y 방향 폭 (nm)
        x_offset       : x 방향 중심 오프셋 (nm)
        y_offset       : y 방향 중심 오프셋 (nm)
        two_deg_depth  : 2DEG 깊이 (nm)
        split_shapes   : split gate shapes (교차 검사용)
        do_describe    : True이면 좌표 출력
        do_plot        : True이면 시각화
        do_overlap_check: True이면 split gate와 교차 검사

    반환:
        TrenchGateStructure
    """
    cfg = TrenchGateConfig(
        x_length=x_length,
        y_width=y_width,
        x_offset=x_offset,
        y_offset=y_offset,
        two_deg_depth=two_deg_depth,
    )

    tg = TrenchGateStructure(cfg)
    tg.build_shapes()

    # Split gate와 교차 검사
    if do_overlap_check and split_shapes is not None:
        tg.check_overlap_with_splitgate(split_shapes)
        print("[INFO] No overlap detected between trench gate and split gate.")

    if do_describe:
        tg.describe()

    if do_plot:
        tg.plot(split_shapes=split_shapes)

    return tg


# 모듈 단독 실행 시
if __name__ == "__main__":
    pass