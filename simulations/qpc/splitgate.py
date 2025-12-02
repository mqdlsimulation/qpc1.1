# gate.py

from typing import Literal, List

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
    plot_electrostatics_multi_structures,
    show_all_figures,
)


# ---------------------------------------------------------
# 1. 모든 구조(여러 gap 포함)를 정의하는 함수
# ---------------------------------------------------------

def build_gate_structure(
    do_describe: bool = True,
    do_plot: bool = False,
) -> List[GateStructure]:
    """
    split gate geometry들을 정의하는 helper.

    여기에서:
      - gap 리스트를 포함한 모든 geometry 파라미터를 한꺼번에 다룬다.
      - 여러 gap에 대해 여러 GateStructure를 만들어 List로 반환한다.

    좌표계:
      - x축: constriction 폭 방향 (갭 방향)
      - y축: 전도 채널 방향 (채널 길이)
    """

    # ---- 여기에서만 geometry 물리 숫자를 정의 ----
    # 여러 gap 값들 (예: 40~80 nm QPC 폭)
    gap_list = [40, 50, 60, 70, 80]  # nm

    # 일반적인 split gate QPC:
    # - 채널 폭: gap
    # - 채널 길이: gate_length_y (짧게, 예: 80 nm)
    # - 패드는 x 방향으로 매우 넓게 (gate_width_x 크게)
    gate_width_x = 500   # nm, x 방향으로 넓은 패드
    gate_length_y = 80    # nm, 채널 길이 (전도 채널 방향)
    two_deg_depth = 60.0    # nm, 2DEG 깊이

    # DUT 전체 영역 (단순히 그림/시뮬 범위용)
    use_dut_offset = False
    dut_Lx = 10000.0        # nm
    dut_Ly = 1000.0         # nm

    structures: List[GateStructure] = []

    for g in gap_list:
        gs = build_split_gate_structure(
            gap=g,
            gate_width_x=gate_width_x,
            gate_length_y=gate_length_y,
            two_deg_depth=two_deg_depth,
            use_dut_offset=use_dut_offset,
            dut_Lx=dut_Lx,
            dut_Ly=dut_Ly,
            do_describe=do_describe,
            do_plot=do_plot,   # 여러 구조를 만들 때 개별 그림은 꺼둘 수 있음
        )
        structures.append(gs)

    return structures


# ---------------------------------------------------------
# 2. gate voltage 설정: mV 값 + mode만
# ---------------------------------------------------------

# 여기서 split gate에 symmetric 하게 전압을 인가할지 각자 따로 인가할지 모드 설정
def configure_gate_voltages(
    mode: Literal["symmetric", "individual"] = "symmetric",
) -> SplitGateVoltages:
    """
    split gate 두 개에 인가할 전압을 설정.

    - 여기서는 mV 파라미터와 mode만 정하고
    - 실제 0.1 mV 양자화 및 Volt 변환은 voltages_splitgate 모듈에서 처리.
    """
    # ---- 여기에서만 전압 숫자를 정의 ----
    SYMMETRIC_VG_MV: float = -100      # symmetric 모드일 때 양쪽  mV
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
# 3. main: 구조/전압 생성 + 플로팅만 수행 (물리 숫자 X)
# ---------------------------------------------------------

def main() -> None:
    # 1) 여러 구조와 전압 생성
    gate_structs = build_gate_structure(
        do_describe=True,
        do_plot=False,   # geometry 시각화는 필요시 True로
    )
    gate_voltages = configure_gate_voltages(mode="symmetric")

    # 대표 구조 하나만 골라서 전체 2D 전위맵 확인 (예: 첫 번째 gap)
    rep_struct = gate_structs[0]

    # 화면용 축 범위 (여기는 단순 plotting 범위이므로 허용) 위에서 내려다본 figure x, y 축 범위 설정
    xlim_2d = (-500.0, 500.0)   # x 축 보기 범위 [nm]
    ylim_2d = (-500.0, 500.0)   # y 축 보기 범위 [nm]

    plot_electrostatics_single(
        gate_struct=rep_struct,
        gate_voltages=gate_voltages,
        screened=False,
        nx=1001,
        ny=1001,
        margin_ratio=0.3,
        xlim=xlim_2d,
        ylim=ylim_2d,
        title="Split gate electrostatic potential (φ(x,y))",
    )

    # 2) 여러 구조에 대해 φ(x,0) 비교 (갭이 다른 QPC들), 퍼텐셜 그래프 x축 범위 설정
    xlim_cut = (-300.0, 300.0)

    plot_electrostatics_multi_structures(
        gate_structs=gate_structs,
        gate_voltages=gate_voltages,
        screened=False,
        nx=1001,
        margin_ratio=0.3,
        xlim=xlim_cut,
        title="φ(x, y=0) for multiple split-gate gaps",
    )

    # 모든 figure 동시에 표시
    show_all_figures()


if __name__ == "__main__":
    main()
