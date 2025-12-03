# modules/helper/plot/plot_electrostatics_common.py

"""
plot_electrostatics_common.py

Electrostatics 결과(φ(x, y))를 시각화하는 공통 helper 모듈.

gate 종류(splitgate, trenchgate, 결합 구조 등)에 무관하게 사용 가능한
일반화된 플롯 함수들을 제공합니다.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from modules.electrostatics.electrostatics_core import (
    ElectrostaticsGrid,
    PotentialMap,
    make_uniform_grid,
)


# ---------------------------------------------------------------------
# 1. 2D 전위 맵 플롯 (일반화)
# ---------------------------------------------------------------------

def plot_potential_2d(
    grid: ElectrostaticsGrid,
    phi: np.ndarray,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    show: bool = False,
) -> None:
    """
    2D 전위 맵 φ(x, y)를 colormap으로 플롯.

    파라미터:
        grid  : ElectrostaticsGrid
        phi   : 2D numpy 배열, shape = (Ny, Nx)
        xlim  : x축 표시 범위
        ylim  : y축 표시 범위
        title : 그래프 제목
        cmap  : colormap 이름
        show  : True이면 plt.show() 호출
    """
    X, Y = grid.meshgrid()

    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, phi, shading="auto", cmap=cmap)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    if title is None:
        title = "Electrostatic potential φ(x, y)"
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
# 2. 여러 전위에 대한 y=0 cut 비교 플롯 (일반화)
# ---------------------------------------------------------------------

def plot_potential_cuts_y0(
    grid: ElectrostaticsGrid,
    phi_list: List[np.ndarray],
    labels: List[str],
    *,
    gap_list: Optional[List[float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_gap_lines: bool = True,
    show: bool = False,
) -> None:
    """
    여러 전위 배열에 대해 y=0에서의 φ(x, y=0) cut을 한 figure에 플롯.

    파라미터:
        grid      : ElectrostaticsGrid
        phi_list  : 전위 배열들의 리스트, 각 shape = (Ny, Nx)
        labels    : 각 전위에 대한 레이블 리스트
        gap_list  : 각 전위에 해당하는 gap 값 리스트 (수직선 표시용, optional)
        xlim      : x축 표시 범위
        title     : 그래프 제목
        show_gap_lines : True이면 gap 위치에 수직 점선 표시
        show      : True이면 plt.show() 호출
    """
    fig, ax = plt.subplots()

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_arr = grid.x
    y_arr = grid.y
    idx_y0 = int(np.argmin(np.abs(y_arr - 0.0)))

    for idx, (phi, label) in enumerate(zip(phi_list, labels)):
        phi_cut = phi[idx_y0, :]
        color = color_cycle[idx % len(color_cycle)]

        ax.plot(x_arr, phi_cut, label=label, color=color)

        # gap 위치에 수직 점선 표시
        if show_gap_lines and gap_list is not None and idx < len(gap_list):
            gap_val = gap_list[idx]
            if gap_val is not None:
                half_gap = gap_val / 2.0
                ax.axvline(x=+half_gap, color=color, linestyle='--', alpha=0.5, linewidth=1.0)
                ax.axvline(x=-half_gap, color=color, linestyle='--', alpha=0.5, linewidth=1.0)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("φ(x, y=0) (V)")
    if title is None:
        title = "φ(x, y=0) comparison"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()


# ---------------------------------------------------------------------
# 3. x=0, y=0 두 방향 cut 플롯
# ---------------------------------------------------------------------

def plot_potential_cuts_both(
    grid: ElectrostaticsGrid,
    phi: np.ndarray,
    *,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    단일 전위에 대해 x=0, y=0 두 방향의 cut을 플롯.

    파라미터:
        grid  : ElectrostaticsGrid
        phi   : 2D numpy 배열, shape = (Ny, Nx)
        title : 전체 figure 제목
        show  : True이면 plt.show() 호출
    """
    x_arr = grid.x
    y_arr = grid.y

    # y=0 cut
    idx_y0 = int(np.argmin(np.abs(y_arr - 0.0)))
    phi_y0 = phi[idx_y0, :]

    # x=0 cut
    idx_x0 = int(np.argmin(np.abs(x_arr - 0.0)))
    phi_x0 = phi[:, idx_x0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # φ(x, y=0)
    axes[0].plot(x_arr, phi_y0)
    axes[0].set_xlabel("x (nm)")
    axes[0].set_ylabel("φ(x, y=0) (V)")
    axes[0].set_title("φ(x, y=0)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color="k", linestyle="--", alpha=0.3)

    # φ(x=0, y)
    axes[1].plot(y_arr, phi_x0)
    axes[1].set_xlabel("y (nm)")
    axes[1].set_ylabel("φ(x=0, y) (V)")
    axes[1].set_title("φ(x=0, y)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color="k", linestyle="--", alpha=0.3)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()


# ---------------------------------------------------------------------
# 4. 개별 gate 기여도 비교 플롯
# ---------------------------------------------------------------------

def plot_individual_contributions(
    grid: ElectrostaticsGrid,
    potentials: Dict[str, PotentialMap],
    *,
    cut_direction: str = "y0",
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    여러 gate의 개별 전위 기여도를 비교하는 플롯.

    파라미터:
        grid          : ElectrostaticsGrid
        potentials    : {"label": PotentialMap, ...} 형태의 dict
        cut_direction : "y0" (y=0에서 cut) 또는 "x0" (x=0에서 cut)
        xlim          : 표시 범위
        title         : 그래프 제목
        show          : True이면 plt.show() 호출
    """
    fig, ax = plt.subplots()

    x_arr = grid.x
    y_arr = grid.y

    if cut_direction == "y0":
        idx_cut = int(np.argmin(np.abs(y_arr - 0.0)))
        coord_arr = x_arr
        xlabel = "x (nm)"
        ylabel = "φ(x, y=0) (V)"
    else:  # x0
        idx_cut = int(np.argmin(np.abs(x_arr - 0.0)))
        coord_arr = y_arr
        xlabel = "y (nm)"
        ylabel = "φ(x=0, y) (V)"

    for name, pot_map in potentials.items():
        if cut_direction == "y0":
            phi_cut = pot_map.phi[idx_cut, :]
        else:
            phi_cut = pot_map.phi[:, idx_cut]

        ax.plot(coord_arr, phi_cut, label=name)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = "Individual gate contributions"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)

    if show:
        plt.show()


# ---------------------------------------------------------------------
# 5. Gate geometry top view 플롯
# ---------------------------------------------------------------------

def plot_gate_geometry(
    all_shapes: Dict[str, List[Tuple[float, float]]],
    *,
    colors: Optional[Dict[str, str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Gate geometry를 top view로 플롯.

    파라미터:
        all_shapes : {"gate_name": [(x1,y1), ...], ...} 형태
        colors     : {"gate_name": "color", ...} 형태 (optional)
        xlim, ylim : 표시 범위
        title      : 그래프 제목
        show       : True이면 plt.show() 호출
    """
    fig, ax = plt.subplots()

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (name, verts) in enumerate(all_shapes.items()):
        xs = [v[0] for v in verts] + [verts[0][0]]
        ys = [v[1] for v in verts] + [verts[0][1]]

        if colors is not None and name in colors:
            color = colors[name]
        else:
            color = color_cycle[idx % len(color_cycle)]

        ax.fill(xs, ys, alpha=0.3, color=color, label=name)
        ax.plot(xs, ys, color=color, linewidth=2)

    ax.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if title is None:
        title = "Gate geometry - Top view"
    ax.set_title(title)
    ax.legend(loc="best")

    if show:
        plt.show()


# ---------------------------------------------------------------------
# 6. 모든 figure를 동시에 표시
# ---------------------------------------------------------------------

def show_all_figures() -> None:
    """
    plt.show()를 호출하여 대기 중인 모든 figure를 동시에 표시.
    """
    plt.show()