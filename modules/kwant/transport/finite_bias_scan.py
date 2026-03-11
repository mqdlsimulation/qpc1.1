# modules/kwant/transport/finite_bias_scan.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import kwant

from modules.electrostatics.electrostatics_core import ElectrostaticsGrid
from modules.electrostatics.electrostatics_split_trench import (
    SplitTrenchStructure,
    SplitTrenchVoltages,
)
from modules.gate_voltages.voltages_splitgate import (
    SplitGateVoltages,
    create_splitgate_voltages_from_mV,
)
from modules.gate_voltages.voltages_trenchgate import (
    TrenchGateVoltages,
    create_trenchgate_voltages_from_mV,
)
from modules.kwant.mesa.mesa1_1 import KwantMesaSystem
from modules.kwant.potential.potential_bridge import (
    compute_phi_for_split_trench,
    two_terminal_conductance,
)


# ----------------------------------------------------------------------
# 설정 / 결과 데이터 클래스
# ----------------------------------------------------------------------


@dataclass
class FiniteBiasScanConfig:
    """
    finite-bias conductance map 설정.

    vq_mV        : V_QPC sweep (split gate 좌/우 대칭 전압, 단위 mV)
    vsd_mV       : V_SD sweep (source-drain bias, 단위 mV)
    trench_Vg_mV : Trench gate 전압 (고정), 단위 mV

    energy_center_ev : 에너지 중심 (보통 Fermi 근처, 예: EF_2DEG_EV)
    eta              : E = E_center + eta * V_SD[V] 로 매핑할 때의 계수
                       기본값 0.5  (양쪽 리저버에 ±eV_SD/2 씩 분배되는 단순 모델)
    lead_in, lead_out: 입력/출력 리드 인덱스 (0, 1)
    """

    vq_mV: Sequence[float]
    vsd_mV: Sequence[float]
    trench_Vg_mV: float

    energy_center_ev: float
    eta: float = 0.5

    lead_in: int = 0
    lead_out: int = 1


@dataclass
class FiniteBiasScanResult:
    """
    finite-bias conductance map 결과.

    G_map    : shape = (len(vsd_mV), len(vq_mV)), 단위 e^2/h (kwant transmission)
    vq_mV    : V_QPC 축 (gate 전압, mV)
    vsd_mV   : V_SD 축 (source-drain bias, mV)
    energies : 각 V_SD 에 대응한 energy [eV] 배열 (len(vsd_mV))
    """

    G_map: np.ndarray
    vq_mV: np.ndarray
    vsd_mV: np.ndarray
    energies: np.ndarray

    def shape(self) -> Tuple[int, int]:
        return self.G_map.shape


# ----------------------------------------------------------------------
# Helper: SplitTrenchVoltages 생성
# ----------------------------------------------------------------------


def _make_split_trench_voltages(
    vq_mV: float,
    trench_Vg_mV: float,
) -> SplitTrenchVoltages:
    """
    대칭 split gate (left/right = V_QPC) + 고정 trench gate 전압에서
    SplitTrenchVoltages 객체를 생성.
    """
    # split gate: 좌우 대칭
    split_volts: SplitGateVoltages = create_splitgate_voltages_from_mV(
        mode="symmetric",
        symmetric_Vg_mV=vq_mV,
        V_left_mV=0.0,
        V_right_mV=0.0,
    )

    # trench gate: 단일 전압
    trench_volts: TrenchGateVoltages = create_trenchgate_voltages_from_mV(
        V_trench_mV=trench_Vg_mV
    )

    return SplitTrenchVoltages(
        split_voltages=split_volts,
        trench_voltages=trench_volts,
    )


# ----------------------------------------------------------------------
# Main: finite-bias conductance map 계산
# ----------------------------------------------------------------------


def run_finite_bias_scan(
    grid: ElectrostaticsGrid,
    structure: SplitTrenchStructure,
    km_system: KwantMesaSystem,
    config: FiniteBiasScanConfig,
) -> FiniteBiasScanResult:
    """
    주어진 grid / gate 구조 / mesa 시스템 / 스캔 설정에서
    finite-bias conductance map G(V_QPC, V_SD)를 계산.

    계산 순서:
        1) kwant 시스템 finalize (geometry 1번만)
        2) 각 V_QPC 에 대해:
            - electrostatics로 φ(x,y) 계산 → U(x,y) [eV]
            - kwant params('pot') 업데이트
            - 각 V_SD 에 대해:
                * E = E_center + eta * V_SD[V]
                * two_terminal_conductance(...) 호출
    """
    vq_arr = np.array(config.vq_mV, dtype=float)
    vsd_arr = np.array(config.vsd_mV, dtype=float)

    n_q = vq_arr.size
    n_sd = vsd_arr.size

    # 결과 배열 (V_SD index, V_QPC index)
    G_map = np.zeros((n_sd, n_q), dtype=float)

    # V_SD [mV] → [V] → energy shift [eV]
    vsd_V = vsd_arr * 1e-3
    dE = config.eta * vsd_V  # e=1 (eV = e·V) 단위계 가정
    energies = config.energy_center_ev + dE

    # kwant 시스템 finalize (geometry 재사용)
    fsys: kwant.system.FiniteSystem = km_system.finalize()

    # kwant params 중 geometry/밴드 파라미터(전압과 무관한 부분)
    base_params = {
        "c": km_system.c_ev,  # onsite 기준 에너지
        "B": 0.0,             # 자기장 확장 시 여기에 반영
    }

    # ----- V_QPC sweep -----
    for iq, vq in enumerate(vq_arr):
        # 1) 현재 V_QPC, 고정 trench V_G 로 electrostatics 계산
        st_voltages = _make_split_trench_voltages(
            vq_mV=vq,
            trench_Vg_mV=config.trench_Vg_mV,
        )

        state = compute_phi_for_split_trench(
            grid=grid,
            structure=structure,
            voltages=st_voltages,
            screened=False,
            label=f"split_trench_VQ{vq:.1f}mV",
        )

        # 2) 이 상태(state)를 고정하고 energy만 바꿔가며 V_SD sweep
        for isd, E in enumerate(energies):
            G = two_terminal_conductance(
                finalized_system=fsys,
                state=state,
                energy=E,
                lead_in=config.lead_in,
                lead_out=config.lead_out,
                base_params=base_params,
            )
            G_map[isd, iq] = G

    return FiniteBiasScanResult(
        G_map=G_map,
        vq_mV=vq_arr,
        vsd_mV=vsd_arr,
        energies=energies,
    )
