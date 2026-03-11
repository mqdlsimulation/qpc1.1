# modules/electrostatics/electrostatics_trenchgate.py

"""
electrostatics_trenchgate.py

Trench gate 구조가 2DEG 평면에서 만들어내는 전위를
Davies 폴리곤 공식을 사용해 계산하는 모듈.

역할:
- TrenchGateStructure (trench gate geometry)
- TrenchGateVoltages (trench gate 전압)
- ElectrostaticsGrid (x, y, depth_d)

를 입력받아, 2DEG 평면에서의 전위 φ(x, y)를 계산하여 PotentialMap으로 반환.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from modules.electrostatics.electrostatics_core import (
    ElectrostaticsGrid,
    PotentialMap,
    apply_linear_screening,
)
from modules.electrostatics.electrostatics_splitgate import polygon_potential_basis_1V
from modules.gate_structures.trenchgate import TrenchGateStructure
from modules.gate_voltages.voltages_trenchgate import TrenchGateVoltages

Point = Tuple[float, float]


def compute_trenchgate_potential(
    gate_struct: TrenchGateStructure,
    gate_voltages: TrenchGateVoltages,
    grid: ElectrostaticsGrid,
    screened: bool = False,
    label: str = "trenchgate",
) -> PotentialMap:
    """
    Trench gate geometry + voltage를 사용해 2DEG 평면에서의 전위 φ(x, y)를 계산.

    파라미터:
        gate_struct   : TrenchGateStructure
                        내부에 shapes: {"trench": List[(x, y)]} 가 있어야 함
        gate_voltages : TrenchGateVoltages
                        - V_trench (Volt)
        grid          : ElectrostaticsGrid
        screened      : True이면 screening 적용 (현재 stub)
        label         : PotentialMap에 붙일 이름

    반환:
        PotentialMap(phi=φ_total(x, y), label=label)
    """
    # TrenchGateStructure가 가진 polygon 정보 추출
    shapes: Dict[str, List[Point]] = gate_struct.shapes

    if "trench" not in shapes:
        raise KeyError("gate_struct.shapes must contain 'trench' polygon.")

    trench_poly = shapes["trench"]

    # 1V 기준 basis 계산 (기존 splitgate 모듈의 함수 재사용)
    phi_trench_1V = polygon_potential_basis_1V(trench_poly, grid)

    # 현재 인가된 전압 (Volt 단위)
    V_trench = gate_voltages.get_voltage()

    # 선형 결합: φ_total = V_trench * φ_trench^(1V)
    phi_total = V_trench * phi_trench_1V

    # (옵션) Screening 적용
    if screened:
        phi_total = apply_linear_screening(phi_total)

    return PotentialMap(phi=phi_total, label=label)


def compute_trenchgate_basis(
    gate_struct: TrenchGateStructure,
    grid: ElectrostaticsGrid,
) -> Dict[str, np.ndarray]:
    """
    Trench gate의 1V 기준 전위 basis를 계산하여 반환.

    다른 gate들과의 선형 결합에 사용할 수 있도록
    basis만 따로 계산하는 함수.

    파라미터:
        gate_struct : TrenchGateStructure
        grid        : ElectrostaticsGrid

    반환:
        {"trench": φ_trench^(1V)} 형태의 dict
    """
    shapes: Dict[str, List[Point]] = gate_struct.shapes

    if "trench" not in shapes:
        raise KeyError("gate_struct.shapes must contain 'trench' polygon.")

    trench_poly = shapes["trench"]
    phi_trench_1V = polygon_potential_basis_1V(trench_poly, grid)

    return {"trench": phi_trench_1V}