# /modules/gate_structures/splitgate.py

"""
splitgate.py

Split gate geometry generator + 기본적인 시각화 도구.

[사용법 요약]

실행 코드에서:

    from modules.gate_structures.splitgate import (
        GateStructureConfig,
        GateStructure,
        run_default_gate_structure,
    )

    # 2) 직접 파라미터를 주고 싶을 때
    cfg = GateStructureConfig(
        gap=100.0,
        gate_width_x=200.0,
        gate_length_y=400.0,
        two_deg_depth=35.0,
        use_dut_offset=False,
        dut_Lx=1000.0,
        dut_Ly=1000.0,
    )
    gs = GateStructure(cfg)
    gs.build_shapes()
    gs.describe()
    gs.plot()

- 좌표계는 항상 "갭 중심이 (0, 0)" 인 로컬 좌표로 정의한 뒤,
  필요하면 DUT 중앙으로 평행 이동합니다.
- 기본은 좌우 대칭(symmetric) split gate입니다.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Optional

Point = Tuple[float, float]


# ---------------- 기본 split gate 파라미터 ---------------- #

@dataclass
class SplitGateParams:
    """
    Split gate의 기본 파라미터 (geometry-only).

    gap          : 게이트 사이의 전체 갭 (2a), 단위 nm
    gate_width_x : 게이트의 x 방향 폭, 단위 nm
    gate_length_y: 게이트의 y 방향 길이 (채널 방향, y축), 단위 nm
    """
    gap: float
    gate_width_x: float
    gate_length_y: float
    two_deg_depth: float | None = None
    use_dut_offset: bool = False
    dut_Lx: float = 1000.0
    dut_Ly: float = 1000.0


# ---------------- 유저 친화적인 구조 정의용 Config ---------------- #

@dataclass
class GateStructureConfig:
    """
    Split gate 구조 전체를 정의하는 유저 파라미터.
    geometry + (옵션) DUT offset + 2DEG 깊이까지 포함.
    """
    gap: float            # nm, 게이트 사이 전체 갭 (2a)
    gate_width_x: float   # nm, 각 게이트의 x 방향 폭
    gate_length_y: float  # nm, 각 게이트의 y 방향 길이 (채널 방향, y축)
    two_deg_depth: float | None = None  # nm, 2DEG 깊이 (None이면 표시 안 함)

    # DUT 전체 영역 (옵션: 중앙 정렬용)
    use_dut_offset: bool = False
    dut_Lx: float = 1000.0  # nm
    dut_Ly: float = 1000.0  # nm


# ---------------- Polygon 유틸 ---------------- #

def _rectangle_vertices_centered(center: Point,
                                 width_x: float,
                                 length_y: float) -> List[Point]:
    """
    중심 좌표와 폭/길이로 axis-aligned 직사각형의 꼭짓점을 생성.

    center  : (cx, cy)
    width_x : x 방향 전체 폭
    length_y: y 방향 전체 길이
    """
    cx, cy = center
    hx = width_x / 2.0   # half width in x
    hy = length_y / 2.0  # half length in y

    # 시계 혹은 반시계 방향 순서로 나열
    return [
        (cx - hx, cy - hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
        (cx - hx, cy + hy),
    ]


# ---------------- 기본 split gate geometry 생성 함수 ---------------- #

def make_split_gate_shapes(params: SplitGateParams,
                           mode: Literal["symmetric"] = "symmetric"
                           ) -> Dict[str, List[Point]]:
    """
    내부용 general split gate 생성 함수.

    현재는 mode="symmetric"만 지원하며,
    이후 asymmetric 모드를 추가할 때 이 함수를 확장하면 됩니다.

    반환:
        {
            "left":  [(x1, y1), ..., (x4, y4)],
            "right": [(x1, y1), ..., (x4, y4)],
        }
    """
    if mode != "symmetric":
        raise NotImplementedError(f"mode '{mode}' is not implemented yet.")

    a = params.gap / 2.0            # 중심에서 안쪽 엣지까지 거리
    w = params.gate_width_x
    L = params.gate_length_y

    # 좌/우 게이트 중심의 x 위치:
    # - 왼쪽: 안쪽 엣지 = -a, 바깥 엣지 = -(a + w) → 중심 x = -(a + w/2)
    # - 오른쪽: 안쪽 엣지 = +a, 바깥 엣지 = +(a + w) → 중심 x = +(a + w/2)
    cx_left  = -(a + w / 2.0)
    cx_right = +(a + w / 2.0)
    cy = 0.0  # y 방향 중심은 0

    left_vertices = _rectangle_vertices_centered(
        center=(cx_left, cy),
        width_x=w,
        length_y=L,
    )

    right_vertices = _rectangle_vertices_centered(
        center=(cx_right, cy),
        width_x=w,
        length_y=L,
    )

    return {
        "left": left_vertices,
        "right": right_vertices,
    }


def make_symmetric_split_gate(gap: float,
                              gate_width_x: float,
                              gate_length_y: float
                              ) -> Dict[str, List[Point]]:
    """
    [유저가 직접 쓰게 될 메인 geometry 함수]

    gap, gate_width_x, gate_length_y 세 개만 넣으면
    - 갭 중심이 (0, 0) 인 좌표계에서
    - 좌우 대칭 split gate 형상을 생성합니다.

    파라미터:
        gap          : 게이트 사이 전체 갭 (2a), nm
        gate_width_x : 게이트의 x 방향 폭, nm
        gate_length_y: 게이트의 y 방향 길이 (y축), nm

    반환:
        {
            "left":  [(x1, y1), ..., (x4, y4)],
            "right": [(x1, y1), ..., (x4, y4)],
        }
    """
    params = SplitGateParams(
        gap=gap,
        gate_width_x=gate_width_x,
        gate_length_y=gate_length_y,
    )
    return make_split_gate_shapes(params, mode="symmetric")


def offset_shapes(shapes: Dict[str, List[Point]],
                  offset: Point) -> Dict[str, List[Point]]:
    """
    로컬 좌표(갭 중심 기준)로 정의된 split gate를
    글로벌 DUT 좌표로 평행이동할 때 쓰는 유틸리티.

    shapes : make_symmetric_split_gate(...) 의 반환값
    offset : (dx, dy) 만큼 전체 이동
             (예: DUT 전체 중앙을 (Lx/2, Ly/2)로 맞추고 싶을 때)

    반환:
        동일한 구조의 dict 이지만, 모든 점에 offset이 더해진 상태.
    """
    dx, dy = offset
    out: Dict[str, List[Point]] = {}

    for key, verts in shapes.items():
        shifted = [(x + dx, y + dy) for (x, y) in verts]
        out[key] = shifted

    return out


# ---------------- 시각화 함수 (기존 스타일 유지) ---------------- #

def plot_split_gate_geometry(shapes: Dict[str, List[Point]],
                             two_deg_depth: float | None = None,
                             gate_thickness: float = 5) -> None:
    """
    설정한 split gate geometry를 시각화:
      1) 위에서 본 모습 (Top view, x-y)
      2) 정면에서 본 모습 (Front view, x-z 단면)

    파라미터:
        shapes        : make_symmetric_split_gate(...) 의 반환값
        two_deg_depth : 2DEG 깊이 d (nm). None이면 2DEG 라인은 안 그립니다.
                        z축은 '깊이' 방향으로, z=0이 표면, z>0이 아래쪽(반도체 내부)입니다.
        gate_thickness: 정면에서 그릴 때 게이트의 '두께'를 위한 시각적 파라미터 (nm).
                        실제 물리적 두께와는 상관없이 단순 도식용입니다.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # ---------- 1. Top view: x-y ----------
    fig1, ax1 = plt.subplots()
    for name, verts in shapes.items():
        xs = [v[0] for v in verts] + [verts[0][0]]
        ys = [v[1] for v in verts] + [verts[0][1]]
        ax1.plot(xs, ys, label=name)

    # 갭 중심 표시
    ax1.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)

    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    ax1.set_title("Split gate - Top view (x-y)")
    ax1.legend(loc="best")

    # ---------- 2. Front view: x-z 단면 ----------
    # x 범위 계산
    all_xs: List[float] = []
    for verts in shapes.values():
        all_xs.extend(v[0] for v in verts)

    x_min = min(all_xs)
    x_max = max(all_xs)

    fig2, ax2 = plt.subplots()

    # 게이트: z = 0 ~ gate_thickness (표면 위쪽/금속 영역)
    for name, verts in shapes.items():
        xs = [v[0] for v in verts]
        gate_x_min = min(xs)
        gate_x_max = max(xs)
        width_x = gate_x_max - gate_x_min

        rect = Rectangle(
            (gate_x_min, 0.0),   # (x, z) 좌표, 표면이 z=0
            width_x,
            gate_thickness,
            alpha=0.4,
            label=f"{name} gate",
        )
        ax2.add_patch(rect)

    # 2DEG 깊이 표시 (z > 0 방향으로 내부)
    if two_deg_depth is not None:
        ax2.axhline(two_deg_depth, color="k", linestyle="--", label="2DEG")

    # x축 범위
    ax2.set_xlim(x_min - 0.1 * abs(x_max - x_min),
                 x_max + 0.1 * abs(x_max - x_min))

    # z축 범위: 위는 gate, 아래는 더 깊은 내부
    z_min = -gate_thickness * 0.2
    z_max = (two_deg_depth if two_deg_depth is not None else gate_thickness * 2.0) * 1.2
    ax2.set_ylim(z_min, z_max)

    # 깊이 방향이 아래로 내려가도록 (z 증가 = 더 깊이)
    ax2.invert_yaxis()

    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("z (nm)")
    ax2.set_title("Split gate - Front view (x-z)")
    ax2.legend(loc="best")

    plt.show()


# ---------------- GateStructure 클래스 ---------------- #

class GateStructure:
    """
    Split gate geometry를 생성하고, 좌표 출력 및 시각화를 담당하는 클래스.
    """

    def __init__(self, config: GateStructureConfig) -> None:
        self.config = config
        self.shapes_local: Dict[str, List[Point]] | None = None
        self.shapes: Dict[str, List[Point]] | None = None

    def build_shapes(self) -> None:
        """갭 중심 (0,0)을 기준으로 split gate geometry 생성 후, 필요하면 DUT 중심으로 이동."""
        # 1) 로컬 좌표(갭 중심 기준)에서 symmetric split gate 생성
        self.shapes_local = make_symmetric_split_gate(
            gap=self.config.gap,
            gate_width_x=self.config.gate_width_x,
            gate_length_y=self.config.gate_length_y,
        )

        # 2) DUT 중앙으로 평행 이동 (옵션)
        if self.config.use_dut_offset:
            cx = self.config.dut_Lx / 2.0
            cy = self.config.dut_Ly / 2.0
            self.shapes = offset_shapes(self.shapes_local, offset=(cx, cy))
            print(f"[INFO] Shapes offset to DUT center at ({cx}, {cy}) nm.")
        else:
            self.shapes = self.shapes_local

    def describe(self) -> None:
        """현재 geometry의 꼭짓점 좌표를 텍스트로 출력."""
        if self.shapes is None:
            raise RuntimeError("Shapes have not been built yet. Call build_shapes() first.")

        print("[INFO] Split gate vertex coordinates (x, y) in nm:")
        for name, verts in self.shapes.items():
            print(f"  {name}:")
            for (x, y) in verts:
                print(f"    ({x:8.3f}, {y:8.3f})")
            print()

    def plot(self) -> None:
        """Top view / Front view 시각화 호출."""
        if self.shapes is None:
            raise RuntimeError("Shapes have not been built yet. Call build_shapes() first.")

        plot_split_gate_geometry(
            shapes=self.shapes,
            two_deg_depth=self.config.two_deg_depth,
        )


def build_split_gate_structure(
    gap: float,
    gate_width_x: float,
    gate_length_y: float,
    two_deg_depth: float | None = None,
    use_dut_offset: bool = False,
    dut_Lx: float = 1000.0,
    dut_Ly: float = 1000.0,
    do_describe: bool = True,
    do_plot: bool = True,
) -> GateStructure:
    """
    숫자 파라미터만 받아서:
      - GateStructureConfig 생성
      - GateStructure 생성
      - build_shapes()
      - (옵션) describe(), plot()
    를 모두 처리하는 helper.

    gate.py에서는 gap, gate_width_x, gate_length_y 등 숫자와
    do_describe/do_plot만 넘겨주면 된다.
    """
    cfg = GateStructureConfig(
        gap=gap,
        gate_width_x=gate_width_x,
        gate_length_y=gate_length_y,
        two_deg_depth=two_deg_depth,
        use_dut_offset=use_dut_offset,
        dut_Lx=dut_Lx,
        dut_Ly=dut_Ly,
    )

    gs = GateStructure(cfg)
    gs.build_shapes()

    if do_describe:
        gs.describe()
    if do_plot:
        gs.plot()

    return gs

# 모듈 단독 실행 시: 기본 데모만 실행
if __name__ == "__main__":
    pass
