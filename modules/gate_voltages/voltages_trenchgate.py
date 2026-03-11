# modules/gate_voltages/voltages_trenchgate.py

"""
voltages_trenchgate.py

Trench gate에 인가되는 전압을 관리하는 모듈.

역할:
- Trench gate에 인가할 전압 값 관리 (V_trench)
- mV 입력 → 0.1 mV 양자화 → V 변환
- 전위 basis와 선형 결합하여 2DEG 전위 계산 지원
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

# ---- 전압 양자화 설정: 최소 단위 0.1 mV ----
VOLTAGE_STEP_MV: float = 0.1


def _quantize_mV(value_mV: float) -> float:
    """
    입력된 전압(mV)을 VOLTAGE_STEP_MV 단위로 양자화.
    """
    step = VOLTAGE_STEP_MV
    return round(value_mV / step) * step


@dataclass
class TrenchGateVoltagesConfig:
    """
    Trench gate에 인가할 전압 설정.

    V_trench : trench gate에 인가되는 전압 [V]
    """
    V_trench: float = 0.0


class TrenchGateVoltages:
    """
    Trench gate 전압 상태를 관리하는 클래스.
    """

    def __init__(self, config: TrenchGateVoltagesConfig | None = None) -> None:
        if config is None:
            config = TrenchGateVoltagesConfig()
        self.V_trench: float = config.V_trench

    # --------- 전압 읽기/쓰기 --------- #

    def as_dict(self) -> Dict[str, float]:
        """현재 게이트 전압을 dict 형태로 반환."""
        return {"trench": self.V_trench}

    def set_from_dict(self, voltages: Dict[str, float]) -> None:
        """dict로부터 전압 설정."""
        if "trench" in voltages:
            self.V_trench = float(voltages["trench"])

    def set_voltage(self, V: float) -> None:
        """Trench gate 전압 설정."""
        self.V_trench = float(V)

    def get_voltage(self) -> float:
        """현재 trench gate 전압 반환."""
        return self.V_trench

    # --------- 전위 basis와의 결합 --------- #

    def combine_with_basis(
        self,
        basis: Dict[str, np.ndarray],
        extra_offset: float = 0.0,
    ) -> np.ndarray:
        """
        미리 계산된 1 V 기준 전위 basis와 현재 전압을 선형 결합해
        2DEG 평면의 전위 φ(x, y)를 반환.

        파라미터:
            basis: {"trench": np.ndarray} 형태, 1V 기준 전위 맵
            extra_offset: 추가 오프셋 전위 (V)

        반환:
            φ_total = V_trench * φ_trench^(1V) + extra_offset
        """
        if "trench" not in basis:
            raise KeyError("basis dict must contain 'trench' key.")

        phi_trench = np.asarray(basis["trench"])

        phi_total = self.V_trench * phi_trench + extra_offset

        return phi_total


# --------- mV 단위로부터 TrenchGateVoltages 생성하는 helper --------- #

def create_trenchgate_voltages_from_mV(V_trench_mV: float) -> TrenchGateVoltages:
    """
    mV 단위로 입력된 전압을 0.1 mV 단위로 양자화한 뒤,
    TrenchGateVoltages를 생성한다.

    파라미터:
        V_trench_mV: trench gate 전압 (mV)

    반환:
        TrenchGateVoltages
    """
    V_q_mV = _quantize_mV(V_trench_mV)
    V_V = V_q_mV * 1e-3  # mV -> V

    cfg = TrenchGateVoltagesConfig(V_trench=V_V)
    volts = TrenchGateVoltages(cfg)

    return volts


# 모듈 단독 실행 시
if __name__ == "__main__":
    pass