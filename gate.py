# gate.py

"""
게이트 구조 + 게이트 전압 설정 스크립트.

요구사항:
- gate.py에서는 geometry / voltage에 대해
  - describe/plot 여부
  - 파라미터 값 (gap, gate width, Vg 등)
  만 설정.
- 실제 작업(객체 생성, build, 전압 양자화, 모드 분기 등)은
  각 모듈(splitgate.py, voltages_splitgate.py)에서 진행.
"""

from typing import Literal

# 1) gate structure helper import
from modules.gate_structures.splitgate import (
    build_split_gate_structure,
    GateStructure,  # 타입 힌트용
)

# 2) voltages helper import
from modules.gate_voltages.voltages_splitgate import (
    SplitGateVoltages,
    create_splitgate_voltages_from_mV,
)

# electrostatics 모듈
from modules.electrostatics.electrostatics_core import (
    make_uniform_grid,
    ElectrostaticsGrid,
)
from modules.electrostatics.electrostatics_splitgate import (
    compute_splitgate_potential,
)


# ---------------------------------------------------------
# 1. gate structure 설정: 숫자 + describe/plot 여부만
# ---------------------------------------------------------

def build_gate_structure(
    do_describe: bool = True,
    do_plot: bool = True,
) -> GateStructure:
    """
    split gate 구조를 정의한다.

    - 여기서는 geometry 숫자와 do_describe/do_plot만 지정.
    - 실제 GateStructure 생성 및 build/describe/plot은
      splitgate 모듈의 build_split_gate_structure가 처리.
    """
    gap = 100.0           # nm
    gate_width_x = 200.0  # nm
    gate_length_y = 400.0 # nm
    two_deg_depth = 35.0  # nm

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
    split gate 두 개에 인가할 전압을 설정한다.

    - gate.py에서는 전압(mV) 파라미터와 mode만 지정.
    - 실제 0.1 mV 양자화, Volt 변환, 모드 분기는
      voltages_splitgate 모듈의 create_splitgate_voltages_from_mV가 처리.
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
    # 여기서 굳이 print 할 필요는 없음 → 반환만
    return volts


# ---------------------------------------------------------
# 3. electrostatics / kwant stub (나중에 실제 구현)
# ---------------------------------------------------------

def run_electrostatics(gate_struct: GateStructure,
                       gate_voltages: SplitGateVoltages) -> None:
    print("[STUB] run_electrostatics()는 나중에 electrostatics 모듈과 연동 예정입니다.")


def run_kwant() -> None:
    print("[STUB] run_kwant()는 나중에 Kwant 모듈과 연동 예정입니다.")


# ---------------------------------------------------------
# 4. main: 호출 순서만 정의
# ---------------------------------------------------------

def main() -> None:
    gate_struct = build_gate_structure()          # describe, plot 기본 True
    gate_voltages = configure_gate_voltages()     # 기본 mode="symmetric"
    run_electrostatics(gate_struct, gate_voltages)
    run_kwant()


if __name__ == "__main__":
    main()
