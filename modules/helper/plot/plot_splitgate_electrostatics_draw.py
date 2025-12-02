# modules/helper/plot/plot_electrostatics_draw.py

"""
plot_electrostatics_draw.py

Electrostatics 결과(φ(x, y))를 시각화하는 helper 모듈.

- plot_electrostatics_single:
    단일 split gate 구조 + 전압에 대한 2D 전위 맵 플롯
- plot_electrostatics_gap_sweep:
    여러 gap 값에 대해 y=0 cut에서 φ(x, 0)을 한 figure에 겹쳐 그리는 함수

geometry/voltages에 대한 숫자 설정은 gate.py에서 GateStructure/Voltages를 만들 때만 하고,
여기서는 GateStructure 내부의 config를 그대로 읽어서 사용한다.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from modules.electrostatics.electrostatics_core import (
    make_uniform_grid,
    ElectrostaticsGrid,
)
from modules.electrostatics.electrostatics_splitgate import (
    compute_splitgate_potential,
)
from modules.gate_structures.splitgate import (
    GateStructure,
    build_split_gate_structure,
)
from modules.gate_voltages.voltages_splitgate import SplitGateVoltages


# ---------------------------------------------------------------------
# 내부 유틸: GateStructure를 기준으로 grid 범위 잡기
# ---------------------------------------------------------------------

def _make_grid_around_gate(
    gate_struct: GateStructure,
    nx: int = 201,
    ny: int = 201,
    margin_ratio: float = 0.3,
) -> ElectrostaticsGrid:
    """
    GateStructure에 포함된 polygon vertex 범위를 기준으로
    grid 영역을 잡는 helper.

    margin_ratio: 게이트 영역의 크기에 대해 바깥쪽으로 추가할 margin 비율.
    """
    shapes = gate_struct.shapes  # type: ignore[attr-defined]
    xs: List[float] = []
    ys: List[float] = []

    for verts in shapes.values():
        xs.extend(v[0] for v in verts)
        ys.extend(v[1] for v in verts)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = x_max - x_min
    dy = y_max - y_min

    x_min -= dx * margin_ratio
    x_max += dx * margin_ratio
    y_min -= dy * margin_ratio
    y_max += dy * margin_ratio

    # GateStructure 내부에 2DEG 깊이 정보가 있으면 사용
    try:
        cfg = gate_struct.config  # type: ignore[attr-defined]
        d = float(cfg.two_deg_depth)
    except Exception:
        d = 35.0  # fallback

    grid = make_uniform_grid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        nx=nx,
        ny=ny,
        depth_d=d,
    )
    return grid


# ---------------------------------------------------------------------
# 1. 단일 설정에 대한 2D 전위 맵 플롯
# ---------------------------------------------------------------------

def plot_electrostatics_single(
    gate_struct: GateStructure,
    gate_voltages: SplitGateVoltages,
    *,
    screened: bool = False,
    nx: int = 201,
    ny: int = 201,
    margin_ratio: float = 0.3,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    주어진 gate_struct + gate_voltages 에 대해
    2DEG 평면 전위 φ(x, y)를 계산하고 2D colormap으로 플롯.

    gate.py에서는:
      - gate_struct, gate_voltages 생성
      - xlim, ylim, nx, ny, margin_ratio만 지정 후 이 함수를 호출.
    """
    grid = _make_grid_around_gate(
        gate_struct=gate_struct,
        nx=nx,
        ny=ny,
        margin_ratio=margin_ratio,
    )

    pot_map = compute_splitgate_potential(
        gate_struct=gate_struct,
        gate_voltages=gate_voltages,
        grid=grid,
        screened=screened,
        label="splitgate",
    )

    X, Y = grid.meshgrid()
    phi = pot_map.phi

    fig, ax = plt.subplots()
    im = ax.pcolormesh(
        X,
        Y,
        phi,
        shading="auto",
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    if title is None:
        title = "Electrostatic potential in 2DEG (split gate)"
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Potential φ (V)")

    plt.show()


# ---------------------------------------------------------------------
# 2. gap sweep: 여러 gap에 대한 φ(x,0) cut을 한 figure에 플롯
# ---------------------------------------------------------------------

def plot_electrostatics_gap_sweep(
    gaps: List[float],
    gate_struct_template: GateStructure,
    gate_voltages: SplitGateVoltages,
    *,
    screened: bool = False,
    nx: int = 201,
    margin_ratio: float = 0.3,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    gap 리스트에 대해:
      - gate_struct_template 의 config를 기준으로
        각 gap에 맞는 GateStructure를 새로 생성
      - 동일 gate_voltages 사용
      - 각 구조에 대해 φ(x, y)를 계산
      - y=0 cut에서 φ(x, 0)을 한 figure에 겹쳐 그린다.

    geometry 숫자(gate_width_x, gate_length_y, two_deg_depth 등)는
    gate_struct_template.config에서만 읽어온다.
    gate.py에서는 gap 리스트와 gate_struct_template만 넘기면 된다.
    """
    # template에서 geometry config 추출 (단일 소스)
    cfg = gate_struct_template.config  # type: ignore[attr-defined]

    gate_width_x = float(cfg.gate_width_x)
    gate_length_y = float(cfg.gate_length_y)
    two_deg_depth = float(cfg.two_deg_depth)
    use_dut_offset = bool(cfg.use_dut_offset)
    dut_Lx = float(cfg.dut_Lx)
    dut_Ly = float(cfg.dut_Ly)

    plt.figure()

    for gap in gaps:
        # 1. geometry (각 gap마다 새 GateStructure)
        gate_struct = build_split_gate_structure(
            gap=gap,
            gate_width_x=gate_width_x,
            gate_length_y=gate_length_y,
            two_deg_depth=two_deg_depth,
            use_dut_offset=use_dut_offset,
            dut_Lx=dut_Lx,
            dut_Ly=dut_Ly,
            do_describe=False,
            do_plot=False,
        )

        # 2. grid & potential
        grid = _make_grid_around_gate(
            gate_struct=gate_struct,
            nx=nx,
            ny=201,             # y 방향은 충분히 커버되도록
            margin_ratio=margin_ratio,
        )
        pot_map = compute_splitgate_potential(
            gate_struct=gate_struct,
            gate_voltages=gate_voltages,
            grid=grid,
            screened=screened,
            label=f"gap={gap}",
        )

        X, Y = grid.meshgrid()
        phi = pot_map.phi

        # y=0 에 가장 가까운 인덱스 선택
        y_arr = grid.y
        idx_y0 = int(np.argmin(np.abs(y_arr - 0.0)))

        x_arr = grid.x
        phi_cut = phi[idx_y0, :]

        plt.plot(x_arr, phi_cut, label=f"gap={gap:.1f} nm")

    if xlim is not None:
        plt.xlim(*xlim)

    plt.xlabel("x (nm)")
    plt.ylabel("φ(x, y=0) (V)")

    if title is None:
        title = "Potential along y=0 for different split gate gaps"
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.show()
