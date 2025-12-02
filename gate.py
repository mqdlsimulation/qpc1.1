# gate.py

from typing import Literal

from modules.gate_structures.splitgate import (
    build_split_gate_structure,
    GateStructure,
)
from modules.gate_voltages.voltages_splitgate import (
    SplitGateVoltages,
    create_splitgate_voltages_from_mV,
)
from modules.helper.plot.plot_splitgate_electrostatics_draw import (
    plot_electrostatics_single,
    plot_electrostatics_gap_sweep,
)


# ---------------------------------------------------------
# 1. gate structure 설정: 숫자 + describe/plot 여부만
# ---------------------------------------------------------

def build_gate_structure(
    do_describe: bool = True,
    do_plot: bool = True,
) -> GateStructure:
    """
    split gate geometry를 정의하는 helper.

    여기에서만 gap, gate_width_x, gate_length_y, two_deg_depth 등을 결정하고
    실제 build/describe/plot은 splitgate 모듈이 처리한다.
    """
    gap = 100.0           # nm
    gate_width_x = 200.0  # nm
    gate_length_y = 400.0 # nm
    two_deg_depth = 60  # nm

    use_dut_offset = False
    dut_Lx = 1000.0
    dut_Ly = 1000.0

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
    return gs


# ---------------------------------------------------------
# 2. gate voltage 설정: mV 값 + mode만
# ---------------------------------------------------------

def configure_gate_voltages(
    mode: Literal["symmetric", "individual"] = "symmetric",
) -> SplitGateVoltages:
    """
    split gate 두 개에 인가할 전압을 설정.

    - 여기서는 mV 파라미터와 mode만 정하고
    - 실제 0.1 mV 양자화 및 Volt 변환은 voltages_splitgate 모듈에서 처리.
    """
    SYMMETRIC_VG_MV: float = -800.0      # symmetric 모드일 때 양쪽 -0.8 V
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
# 3. main: geometry + voltages + plotting 흐름만 정의
# ---------------------------------------------------------

def main() -> None:
    # 1) geometry / voltages 생성
    gate_struct = build_gate_structure(do_describe=True, do_plot=True)
    gate_voltages = configure_gate_voltages(mode="symmetric")

    # 2) 2D 전위 맵 figure 축 범위 설정 (필요시 조절)
    xlim_2d = (-500.0, 500.0)   # x 축 범위 [nm]
    ylim_2d = (-500.0, 500.0)   # y 축 범위 [nm]

    plot_electrostatics_single(
        gate_struct=gate_struct,
        gate_voltages=gate_voltages,
        screened=False,
        nx=201,
        ny=201,
        margin_ratio=0.3,
        xlim=xlim_2d,
        ylim=ylim_2d,
        title="Split gate electrostatic potential (φ(x,y))",
    )

    # 3) gap sweep: 여러 gap에 대한 φ(x,0) cut 비교 figure
    gaps = [80.0, 100.0, 120.0]  # nm, 필요에 따라 수정
    xlim_cut = (-400.0, 400.0)

    plot_electrostatics_gap_sweep(
        gaps=gaps,
        gate_struct_template=gate_struct,
        gate_voltages=gate_voltages,
        screened=False,
        nx=201,
        margin_ratio=0.3,
        xlim=xlim_cut,
        title="φ(x, y=0) for different split gate gaps",
    )


if __name__ == "__main__":
    main()
