# simulations/qpc/simulation_splitgate.py

"""
simulation_splitgate.py

Split gate 구조의 2DEG 전위 시뮬레이션.

좌표계:
- x축: constriction 폭 방향 (gap 방향)
- y축: 전도 채널 방향
"""

from typing import Literal, List, Tuple, Optional

import numpy as np

from modules.gate_structures.splitgate import (
    build_split_gate_structure,
    GateStructure,
)
from modules.gate_voltages.voltages_splitgate import (
    SplitGateVoltages,
    create_splitgate_voltages_from_mV,
)
from modules.electrostatics.electrostatics_core import (
    ElectrostaticsGrid,
    make_uniform_grid,
)
from modules.electrostatics.electrostatics_splitgate import (
    compute_splitgate_potential,
)
from modules.helper.plot.plot_electrostatics_common import (
    plot_potential_2d,
    plot_potential_cuts_y0,
    plot_gate_geometry,
    show_all_figures,
)


# ---------------------------------------------------------
# 1. Split gate 구조 생성
# ---------------------------------------------------------

def build_splitgates(
    gap_list: List[float],
    do_describe: bool = False,
    do_plot: bool = False,
) -> List[GateStructure]:
    """
    여러 gap에 대한 split gate 구조 리스트 생성.

    좌표계:
      - x축: constriction 폭 방향 (갭 방향)
      - y축: 전도 채널 방향 (채널 길이)
    """

    # ---- geometry 파라미터 ----
    gate_width_x = 500.0    # nm, x 방향 폭
    gate_length_y = 80.0    # nm, 채널 길이 (y 방향)
    two_deg_depth = 60.0    # nm, 2DEG 깊이

    use_dut_offset = False
    dut_Lx = 10000.0
    dut_Ly = 1000.0

    structures: List[GateStructure] = []

    for gap in gap_list:
        gs = build_split_gate_structure(
            gap=gap,
            gate_width_x=gate_width_x,
            gate_length_y=gate_length_y,
            two_deg_depth=two_deg_depth,
            use_dut_offset=use_dut_offset,
            dut_Lx=dut_Lx,
            dut_Ly=dut_Ly,
            do_describe=do_describe,
            do_plot=do_plot,
        )
        structures.append(gs)

    return structures


# ---------------------------------------------------------
# 2. 전압 설정
# ---------------------------------------------------------

def configure_voltages(
    mode: Literal["symmetric", "individual"] = "symmetric",
) -> SplitGateVoltages:
    """
    Split gate 전압 설정.
    """
    # ---- 전압 파라미터 (mV) ----
    SYMMETRIC_VG_MV: float = -100.0
    INDIVIDUAL_V_LEFT_MV: float = -800.0
    INDIVIDUAL_V_RIGHT_MV: float = -750.0

    volts = create_splitgate_voltages_from_mV(
        mode=mode,
        symmetric_Vg_mV=SYMMETRIC_VG_MV,
        V_left_mV=INDIVIDUAL_V_LEFT_MV,
        V_right_mV=INDIVIDUAL_V_RIGHT_MV,
    )
    return volts


# ---------------------------------------------------------
# 3. Grid 생성
# ---------------------------------------------------------

def make_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    depth_d: float,
    nx: int = 401,
    ny: int = 401,
) -> ElectrostaticsGrid:
    """2DEG 평면의 계산 grid 생성."""
    return make_uniform_grid(
        x_min=x_range[0],
        x_max=x_range[1],
        y_min=y_range[0],
        y_max=y_range[1],
        nx=nx,
        ny=ny,
        depth_d=depth_d,
    )


# ---------------------------------------------------------
# 4. Main
# ---------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Split gate simulation")
    print("=" * 60)

    # ========== 파라미터 정의 (한 번만) ==========
    gap_list = [40.0, 80.0, 240.0, 360.0]  # nm

    x_range = (-500.0, 500.0)  # grid 계산 범위
    y_range = (-500.0, 500.0)

    xlim_2d = (-500.0, 500.0)  # 2D 맵 표시 범위
    ylim_2d = (-500.0, 500.0)

    xlim_cut = (-200.0, 200.0)  # cut 그래프 표시 범위

    nx, ny = 401, 401
    depth_d = 60.0

    # ========== 1) 구조 생성 ==========
    gate_structs = build_splitgates(
        gap_list=gap_list,
        do_describe=True,
        do_plot=False,
    )

    # ========== 2) 전압 설정 ==========
    gate_voltages = configure_voltages(mode="symmetric")
    print(f"\n[INFO] Gate voltage: {gate_voltages.V_left * 1000:.1f} mV (symmetric)")

    # ========== 3) Grid 생성 ==========
    grid = make_grid(
        x_range=x_range,
        y_range=y_range,
        nx=nx,
        ny=ny,
        depth_d=depth_d,
    )

    # ========== 4) 전위 계산 ==========
    print("\n[INFO] Computing potentials...")

    phi_list: List[np.ndarray] = []
    labels: List[str] = []

    for gs in gate_structs:
        pot_map = compute_splitgate_potential(
            gate_struct=gs,
            gate_voltages=gate_voltages,
            grid=grid,
            screened=False,
        )
        phi_list.append(pot_map.phi)
        labels.append(f"gap={gs.config.gap:.0f} nm")

    print("[INFO] Computation complete.")

    # ========== 5) 시각화 ==========

    # 대표 구조의 gate geometry
    rep_struct = gate_structs[0]
    plot_gate_geometry(
        all_shapes=rep_struct.shapes,
        title=f"Split gate geometry (gap={rep_struct.config.gap:.0f} nm)",
    )

    # 대표 구조의 2D 전위 맵
    plot_potential_2d(
        grid=grid,
        phi=phi_list[0],
        xlim=xlim_2d,
        ylim=ylim_2d,
        title=f"Split gate φ(x,y), gap={rep_struct.config.gap:.0f} nm",
    )

    # 여러 gap에 대한 φ(x, y=0) 비교
    plot_potential_cuts_y0(
        grid=grid,
        phi_list=phi_list,
        labels=labels,
        gap_list=gap_list,
        xlim=xlim_cut,
        title="φ(x, y=0) for multiple split-gate gaps",
        show_gap_lines=True,
    )

    # 모든 figure 동시에 표시
    show_all_figures()


if __name__ == "__main__":
    main()