# /modules/gate_voltages/voltages_splitgate.py

"""
voltages_splitgate.py

Split gate 구조에 인가되는 게이트 전압을 관리하는 모듈.

역할:
- 왼쪽/오른쪽 split gate에 인가할 전압 값을 관리 (V_left, V_right)
- 대칭(symmetric), 반대칭(antisymmetric) 등 전압 패턴을 쉽게 설정
- (옵션) 미리 계산된 전위 basis(1V 기준 φ_left, φ_right)를
  현재 전압으로 선형결합해서 2DEG 평면의 φ(x, y)를 만들어주는 유틸리티 제공

주의:
- 여기서는 Kwant 해밀토니안/onsite 함수는 정의하지 않는다.
- Kwant 코드 파일(예: hamiltonian_splitgate.py)에서 이 모듈을 import하여
  combine_with_basis(...) 결과를 onsite potential로 사용하면 된다.
"""

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

GateName = Literal["left", "right"]

# ---- 전압 양자화 설정: 최소 단위 0.1 mV ----
VOLTAGE_STEP_MV: float = 0.1


def _quantize_mV(value_mV: float) -> float:
    """
    입력된 전압(mV)을 VOLTAGE_STEP_MV 단위로 양자화.

    예:
        1.234 mV -> 1.2 mV  (step=0.1 mV)
        -0.05 mV -> 0.0 mV
        -0.08 mV -> -0.1 mV
    """
    step = VOLTAGE_STEP_MV
    return round(value_mV / step) * step


@dataclass
class SplitGateVoltagesConfig:
    """
    Split gate에 인가할 전압 설정.

    V_left, V_right : 각 게이트에 인가되는 전압 [V]
    """
    V_left: float = 0.0
    V_right: float = 0.0


class SplitGateVoltages:
    """
    Split gate 전압 상태를 관리하는 클래스.

    - V_left, V_right 전압을 보관
    - symmetric / antisymmetric 세팅 메서드 제공
    - (옵션) 1 V 기준 전위 basis와 결합해 2DEG 전위를 반환하는 메서드 제공
    """

    def __init__(self, config: SplitGateVoltagesConfig | None = None) -> None:
        if config is None:
            config = SplitGateVoltagesConfig()
        self.V_left: float = config.V_left
        self.V_right: float = config.V_right

    # --------- 전압 읽기/쓰기 --------- #

    def as_dict(self) -> Dict[GateName, float]:
        """현재 게이트 전압들을 dict 형태로 반환."""
        return {
            "left": self.V_left,
            "right": self.V_right,
        }

    def set_from_dict(self, voltages: Dict[GateName, float]) -> None:
        """dict로부터 전압 설정."""
        if "left" in voltages:
            self.V_left = float(voltages["left"])
        if "right" in voltages:
            self.V_right = float(voltages["right"])

    def set_symmetric(self, Vg: float) -> None:
        """
        좌우 게이트에 같은 전압을 인가.
        예: QPC를 양쪽에서 동일하게 잠글 때.
        """
        self.V_left = float(Vg)
        self.V_right = float(Vg)

    def set_antisymmetric(self, V: float) -> None:
        """
        좌우에 반대 부호 전압을 인가.
        (필요하다면 drift/tilt 테스트용)
        예: left=+V, right=-V
        """
        self.V_left = float(+V)
        self.V_right = float(-V)

    def set_individual(self, V_left: float, V_right: float) -> None:
        """왼쪽/오른쪽 전압을 개별적으로 설정."""
        self.V_left = float(V_left)
        self.V_right = float(V_right)

    # --------- 전위 basis와의 결합 --------- #

    def combine_with_basis(
        self,
        basis: Dict[GateName, np.ndarray],
        extra_offset: float = 0.0,
    ) -> np.ndarray:
        """
        미리 계산된 1 V 기준 전위 basis와 현재 전압을 선형 결합해
        2DEG 평면의 전위 φ(x, y)를 반환.
        """
        if "left" not in basis or "right" not in basis:
            raise KeyError("basis dict must contain 'left' and 'right' keys.")

        phi_left = np.asarray(basis["left"])
        phi_right = np.asarray(basis["right"])

        if phi_left.shape != phi_right.shape:
            raise ValueError("basis['left'] and basis['right'] must have the same shape.")

        phi_total = (
            self.V_left * phi_left +
            self.V_right * phi_right +
            extra_offset
        )
        return phi_total


# --------- mV 단위로부터 SplitGateVoltages 생성하는 helper --------- #

def create_symmetric_from_mV(Vg_mV: float) -> SplitGateVoltages:
    """
    mV 단위로 입력된 전압을 0.1 mV 단위로 양자화한 뒤,
    좌우 게이트에 같은 전압을 인가하는 SplitGateVoltages를 생성한다.
    """
    Vg_q_mV = _quantize_mV(Vg_mV)
    Vg_V = Vg_q_mV * 1e-3  # mV -> V

    cfg = SplitGateVoltagesConfig(V_left=Vg_V, V_right=Vg_V)
    volts = SplitGateVoltages(cfg)
    return volts


def create_individual_from_mV(V_left_mV: float, V_right_mV: float) -> SplitGateVoltages:
    """
    왼쪽/오른쪽 게이트 전압을 mV 단위로 입력받아
    각각 0.1 mV 단위로 양자화한 뒤 SplitGateVoltages를 생성한다.
    """
    V_left_q_mV = _quantize_mV(V_left_mV)
    V_right_q_mV = _quantize_mV(V_right_mV)

    V_left_V = V_left_q_mV * 1e-3
    V_right_V = V_right_q_mV * 1e-3

    cfg = SplitGateVoltagesConfig(V_left=V_left_V, V_right=V_right_V)
    volts = SplitGateVoltages(cfg)
    return volts

def create_splitgate_voltages_from_mV(
    mode: Literal["symmetric", "individual"],
    symmetric_Vg_mV: float,
    V_left_mV: float,
    V_right_mV: float,
) -> SplitGateVoltages:
    """
    mode + mV 파라미터들을 받아서:
      - mode="symmetric"  → create_symmetric_from_mV 사용
      - mode="individual" → create_individual_from_mV 사용
    으로 SplitGateVoltages를 생성한다.

    gate.py에서는 mode와 mV 값만 넘겨주면 된다.
    """
    if mode == "symmetric":
        return create_symmetric_from_mV(symmetric_Vg_mV)
    elif mode == "individual":
        return create_individual_from_mV(V_left_mV, V_right_mV)
    else:
        raise ValueError(f"Unknown mode: {mode}")