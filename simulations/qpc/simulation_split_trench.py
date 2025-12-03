# simulations/qpc/simulation_split_trench.py

"""
simulation_split_trench.py

Split gate + Trench gate 결합 구조의 2DEG 전위 시뮬레이션.

좌표계:
- x축: split gate constriction 방향 (gap 방향)
- y축: 전도 채널 방향
- z = 0: 웨이퍼 표면
- z < 0: 2DEG 위치 (z = -depth_d)
"""

from typing import List, Tuple

import numpy as np

from modules.gate_structures.splitgate import (
    build_split_gate_structure,
    GateStructure,
)
from modules.gate_structures.trenchgate import (
    build_trench_gate_structure,
    TrenchGateStructure,
)
from modules.gate_voltages.voltages_splitgate import (
    SplitGateVoltages,
    create_splitgate_voltages_from_mV,
)
from modules.gate_voltages.voltages_trenchgate import (
    TrenchGateVoltages,
    create_trenchgate_voltages_from_mV,
)
from modules.electrostatics.electrostatics_core import (
    ElectrostaticsGrid,
    make_uniform_grid,
)
from modules.electrostatics.electrostatics_split_trench import (
    SplitTrenchStructure,
    SplitTrenchVoltages,
    compute_split_trench_potential,
    compute_individual_potentials,
)
from modules.helper.plot.plot_electrostatics_common import (
    plot_potential_2d,
    plot_potential_cuts_y0,
    plot_potential_cuts_both,
    plot_individual_contributions,
    plot_gate_geometry,
    show_all_figures,
)


# ---------------------------------------------------------
# 1. Split gate 구조 생성
# ---------------------------------------------------------

def build_splitgates(
    gap_list: List[float],
    depth_d: float,
    do_describe: bool = False,
    do_plot: bool = False,
) -> List[GateStructure]:
    """여러 gap에 대한 split gate 구조 리스트 생성."""

    # ---- Split gate geometry 파라미터 ----
    gate_width_x = 500.0    # nm, x 방향 폭
    gate_length_y = 80.0    # nm, y 방향 길이 (채널 방향)

    structures: List[GateStructure] = []

    for gap in gap_list:
        gs = build_split_gate_structure(
            gap=gap,
            gate_width_x=gate_width_x,
            gate_length_y=gate_length_y,
            two_deg_depth=depth_d,
            use_dut_offset=False,
            do_describe=do_describe,
            do_plot=do_plot,
        )
        structures.append(gs)

    return structures


# ---------------------------------------------------------
# 2. Trench gate 구조 생성
# ---------------------------------------------------------

def build_trenchgate(
    split_gates: List[GateStructure],
    depth_d: float,
    do_describe: bool = False,
    do_plot: bool = False,
) -> TrenchGateStructure:
    """
    Trench gate 구조 생성.
    첫 번째 split gate 기준으로 교차 검사 수행.
    """

    # ---- Trench gate geometry 파라미터 ----
    x_length = 20.0       # nm, x 방향 길이 (split gate gap을 가로지름)
    y_width = 500.0       # nm, y 방향 폭
    x_offset = 0.0        # nm
    y_offset = 0.0        # nm

    tg = build_trench_gate_structure(
        x_length=x_length,
        y_width=y_width,
        x_offset=x_offset,
        y_offset=y_offset,
        two_deg_depth=depth_d,
        split_shapes=split_gates[0].shapes,
        do_describe=do_describe,
        do_plot=do_plot,
        do_overlap_check=True,
    )
    return tg


# ---------------------------------------------------------
# 3. 전압 설정
# ---------------------------------------------------------

def configure_voltages() -> SplitTrenchVoltages:
    """Split gate + Trench gate 전압 설정."""

    # ---- 전압 파라미터 (mV) ----
    SPLIT_VG_MV: float = -500.0    # split gate 양쪽 동일 전압(mV)
    TRENCH_VG_MV: float = 1500   # trench gate 전압(mV)

    split_volts = create_splitgate_voltages_from_mV(
        mode="symmetric",
        symmetric_Vg_mV=SPLIT_VG_MV,
        V_left_mV=0.0,
        V_right_mV=0.0,
    )

    trench_volts = create_trenchgate_voltages_from_mV(TRENCH_VG_MV)

    return SplitTrenchVoltages(
        split_voltages=split_volts,
        trench_voltages=trench_volts,
    )


# ---------------------------------------------------------
# 4. Grid 생성
# ---------------------------------------------------------

def make_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    depth_d: float,
    nx: int = 1001,
    ny: int = 1001,
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
# 5. Main
# ---------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Split gate + Trench gate simulation")
    print("=" * 60)

    # ========== 파라미터 정의 (한 번만) ==========
    gap_list = [80, 160, 240, 300]  # nm
    depth_d = 60.0  # nm, 2DEG 깊이 (공통)

    x_range = (-300.0, 300.0)  # grid 계산 범위
    y_range = (-300.0, 300.0)

    xlim_2d = (-250.0, 250.0)  # 2D 맵 표시 범위
    ylim_2d = (-250.0, 250.0)

    xlim_cut = (-200.0, 200.0)  # cut 그래프 표시 범위

    nx, ny = 1001, 1001

    # ========== 1) 구조 생성 ==========
    print("\n[INFO] Building gate structures...")

    split_gates = build_splitgates(
        gap_list=gap_list,
        depth_d=depth_d,
        do_describe=False,
        do_plot=False,
    )

    trench_gate = build_trenchgate(
        split_gates=split_gates,
        depth_d=depth_d,
        do_describe=True,
        do_plot=False,
    )

    # 결합 구조 리스트 생성
    structures: List[SplitTrenchStructure] = []
    for sg in split_gates:
        struct = SplitTrenchStructure(
            split_gate=sg,
            trench_gate=trench_gate,
        )
        structures.append(struct)

    print(f"[INFO] Created {len(structures)} combined structures.")

    # ========== 2) 전압 설정 ==========
    voltages = configure_voltages()
    voltages.describe()

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

    for struct in structures:
        pot_map = compute_split_trench_potential(
            structure=struct,
            voltages=voltages,
            grid=grid,
            screened=False,
        )
        phi_list.append(pot_map.phi)
        labels.append(f"gap={struct.split_gate.config.gap:.0f} nm")

    # 대표 구조의 개별 기여도 계산
    rep_structure = structures[0]
    pot_individual = compute_individual_potentials(
        structure=rep_structure,
        voltages=voltages,
        grid=grid,
        screened=False,
    )

    print("[INFO] Computation complete.")

    # ========== 5) 시각화 ==========

    # Gate geometry (대표 구조)
    all_shapes = rep_structure.get_all_shapes()
    plot_gate_geometry(
        all_shapes=all_shapes,
        colors={"split_left": "C0", "split_right": "C0", "trench": "C1"},
        xlim=(-350, 350),
        ylim=(-350, 350),
        title=f"Gate geometry (gap={rep_structure.split_gate.config.gap:.0f} nm)",
    )

    # 2D 전위 맵 (대표 구조)
    plot_potential_2d(
        grid=grid,
        phi=phi_list[0],
        xlim=xlim_2d,
        ylim=ylim_2d,
        title=f"Combined φ(x,y), gap={rep_structure.split_gate.config.gap:.0f} nm",
    )

    # x=0, y=0 두 방향 cut (대표 구조)
    plot_potential_cuts_both(
        grid=grid,
        phi=phi_list[0],
        title=f"Potential cuts (gap={rep_structure.split_gate.config.gap:.0f} nm)",
    )

    # 여러 gap에 대한 φ(x, y=0) 비교
    plot_potential_cuts_y0(
        grid=grid,
        phi_list=phi_list,
        labels=labels,
        gap_list=gap_list,
        xlim=xlim_cut,
        title="φ(x, y=0) for multiple gaps [Split + Trench]",
        show_gap_lines=True,
    )

    # 개별 gate 기여도 비교 (대표 구조)
    plot_individual_contributions(
        grid=grid,
        potentials=pot_individual,
        cut_direction="y0",
        xlim=xlim_cut,
        title="Individual gate contributions at y=0",
    )

    # 모든 figure 동시에 표시
    show_all_figures()


if __name__ == "__main__":
    main()