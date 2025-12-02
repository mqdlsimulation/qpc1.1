# modules/helper/plot/plot_splitgate_electrostatics_draw.py

"""
plot_electrostatics_draw.py

Electrostatics 결과(φ(x, y))를 시각화하는 helper 모듈.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from modules.electrostatics.electrostatics_core import ElectrostaticsGrid, make_uniform_grid
from modules.electrostatics.electrostatics_splitgate import compute_splitgate_potential
from modules.gate_structures.splitgate import GateStructure
from modules.gate_voltages.voltages_splitgate import SplitGateVoltages


# ---------------------------------------------------------------------
# 내부 유틸: GateStructure를 기준으로 grid 범위 잡기
# ---------------------------------------------------------------------

def _make_grid_around_gate(
    gate_struct: GateStructure,
    nx: int = 201,
    ny: int = 201,
    margin_ratio: float = 0.3,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> ElectrostaticsGrid:
    """
    GateStructure에 포함된 polygon vertex 범위를 기준으로
    grid 영역을 잡는 helper.
    """
    if x_range is not None and y_range is not None:
        x_min, x_max = x_range
        y_min, y_max = y_range
    else:
        shapes = gate_struct.shapes
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

    try:
        cfg = gate_struct.config
        d = float(cfg.two_deg_depth)
    except Exception:
        d = 60.0

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
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    주어진 gate_struct + gate_voltages 에 대해
    2DEG 평면 전위 φ(x, y)를 계산하고 2D colormap으로 플롯.
    """
    grid = _make_grid_around_gate(
        gate_struct=gate_struct,
        nx=nx,
        ny=ny,
        margin_ratio=margin_ratio,
        x_range=x_range,
        y_range=y_range,
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
    im = ax.pcolormesh(X, Y, phi, shading="auto")
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

    if show:
        plt.show()


# ---------------------------------------------------------------------
# 2. 여러 구조에 대한 φ(x,0) cut을 한 figure에 플롯
# ---------------------------------------------------------------------

def plot_electrostatics_multi_structures(
    gate_structs: List[GateStructure],
    gate_voltages: SplitGateVoltages,
    *,
    screened: bool = False,
    nx: int = 201,
    ny: int = 201,
    margin_ratio: float = 0.3,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_gap_lines: bool = True,
    show: bool = False,
) -> None:
    """
    여러 GateStructure들에 대해 각각 φ(x,y)를 계산하고,
    y=0에서 φ(x,0)을 한 figure에 겹쳐서 그리는 함수.

    show_gap_lines: True이면 각 gap 위치에 수직 점선 표시
    """
    fig, ax = plt.subplots()

    # 색상 cycle 가져오기
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, gs in enumerate(gate_structs):
        # gap 값 추출
        try:
            cfg = gs.config
            gap_val = float(cfg.gap)
            label = f"gap={gap_val:.0f} nm"
        except Exception:
            gap_val = None
            label = "structure"

        grid = _make_grid_around_gate(
            gate_struct=gs,
            nx=nx,
            ny=ny,
            margin_ratio=margin_ratio,
            x_range=x_range,
            y_range=y_range,
        )

        pot_map = compute_splitgate_potential(
            gate_struct=gs,
            gate_voltages=gate_voltages,
            grid=grid,
            screened=screened,
            label=label,
        )

        phi = pot_map.phi

        # y = 0 cut
        y_arr = grid.y
        idx_y0 = int(np.argmin(np.abs(y_arr - 0.0)))
        x_arr = grid.x
        phi_cut = phi[idx_y0, :]

        # 현재 색상
        color = color_cycle[idx % len(color_cycle)]

        # potential curve 플롯
        ax.plot(x_arr, phi_cut, label=label, color=color)

        # gap 위치에 수직 점선 표시
        if show_gap_lines and gap_val is not None:
            half_gap = gap_val / 2.0
            ax.axvline(x=+half_gap, color=color, linestyle='--', alpha=0.5, linewidth=1.0)
            ax.axvline(x=-half_gap, color=color, linestyle='--', alpha=0.5, linewidth=1.0)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("φ(x, y=0) (V)")
    if title is None:
        title = "φ(x, y=0) for multiple split-gate structures"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()


# ---------------------------------------------------------------------
# 3. 모든 figure를 동시에 표시
# ---------------------------------------------------------------------

def show_all_figures() -> None:
    """
    plt.show()를 호출하여 대기 중인 모든 figure를 동시에 표시.
    """
    plt.show()