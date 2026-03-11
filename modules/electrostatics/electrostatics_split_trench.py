# modules/electrostatics/electrostatics_split_trench.py

"""
electrostatics_split_trench.py

Split gate + Trench gate 결합 구조가 2DEG 평면에서 만들어내는 전위를
Davies 폴리곤 공식을 사용해 계산하는 모듈.

역할:
- Split gate + Trench gate geometry
- 각 gate의 전압
- ElectrostaticsGrid (x, y, depth_d)

를 입력받아, 2DEG 평면에서의 결합 전위 φ(x, y)를 계산하여 PotentialMap으로 반환.

향후 확장:
- 전하 screening 효과 적용
- gate 간 상호작용 효과
- 기타 물리 규칙 추가
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from modules.electrostatics.electrostatics_core import (
    ElectrostaticsGrid,
    PotentialMap,
    apply_linear_screening,
)
from modules.gate_structures.splitgate import GateStructure
from modules.gate_structures.trenchgate import TrenchGateStructure
from modules.gate_voltages.voltages_splitgate import SplitGateVoltages
from modules.gate_voltages.voltages_trenchgate import TrenchGateVoltages

Point = Tuple[float, float]


# --------------------------------------------------------------------
# Davies polygon formula 기반: 단일 polygon gate의 1 V basis 계산
# --------------------------------------------------------------------

def _cot(x: np.ndarray) -> np.ndarray:
    """
    수치적 안정성을 고려한 cot(x) 계산.
    """
    tan_x = np.tan(x)
    tan_x[tan_x == 0] += 1e-15
    return 1.0 / tan_x


def _polygon_potential_basis_1V(
    vertices: List[Point],
    grid: ElectrostaticsGrid,
) -> np.ndarray:
    """
    Davies 폴리곤 공식을 사용하여,
    "주어진 polygon gate에 1 V가 인가되었을 때"
    2DEG 평면(grid.depth_d)에서의 전위 분포 φ(x, y)를 계산한다.

    파라미터:
        vertices : [(x1, y1), ..., (xN, yN)] 형태의 polygon 꼭짓점 리스트
        grid     : ElectrostaticsGrid (x, y, depth_d)

    반환:
        phi_1V : 2D numpy 배열, shape = (Ny, Nx)
    """
    X, Y = grid.meshgrid()
    z = grid.depth_d

    r = X + 1j * Y
    pn = len(vertices)

    p = np.array(vertices, dtype=float)
    p_c = p[:, 0] + 1j * p[:, 1]

    basis = np.zeros_like(r, dtype=float)

    for m in range(pn):
        p0 = p_c[(m - 1) % pn]
        p1 = p_c[m]
        p2 = p_c[(m + 1) % pn]

        a = -np.angle((r - p1) / (p0 - p1))
        b = np.angle((r - p1) / (p2 - p1))

        dist_sq = np.abs(r - p1) ** 2
        sin_g = z / np.sqrt(dist_sq + z**2)

        cot_a = _cot(a)
        cot_b = _cot(b)

        basis += (
            (np.arctan(cot_a) - np.arctan(sin_g * cot_a))
            + (np.arctan(cot_b) - np.arctan(sin_g * cot_b))
        )

    phi_1V = -basis / (2.0 * np.pi)
    return phi_1V


# --------------------------------------------------------------------
# Combined gate voltages 데이터 클래스
# --------------------------------------------------------------------

@dataclass
class SplitTrenchVoltages:
    """
    Split gate + Trench gate 전압을 함께 관리하는 데이터 클래스.
    """
    split_voltages: SplitGateVoltages
    trench_voltages: TrenchGateVoltages

    def describe(self) -> None:
        """현재 전압 설정을 출력."""
        split_dict = self.split_voltages.as_dict()
        trench_dict = self.trench_voltages.as_dict()
        print("[INFO] Gate voltages:")
        print(f"  Split gate (left):  {split_dict['left']*1000:.1f} mV")
        print(f"  Split gate (right): {split_dict['right']*1000:.1f} mV")
        print(f"  Trench gate:        {trench_dict['trench']*1000:.1f} mV")


# --------------------------------------------------------------------
# Combined gate structures 데이터 클래스
# --------------------------------------------------------------------

@dataclass
class SplitTrenchStructure:
    """
    Split gate + Trench gate 구조를 함께 관리하는 데이터 클래스.
    """
    split_gate: GateStructure
    trench_gate: TrenchGateStructure

    def get_all_shapes(self) -> Dict[str, List[Point]]:
        """모든 gate의 polygon shapes를 하나의 dict로 반환."""
        all_shapes: Dict[str, List[Point]] = {}

        # Split gate shapes
        if self.split_gate.shapes is not None:
            for name, verts in self.split_gate.shapes.items():
                all_shapes[f"split_{name}"] = verts

        # Trench gate shapes
        if self.trench_gate.shapes is not None:
            for name, verts in self.trench_gate.shapes.items():
                all_shapes[name] = verts

        return all_shapes

    def describe(self) -> None:
        """구조 정보 출력."""
        print("[INFO] Combined gate structure:")
        print(f"  Split gate gap: {self.split_gate.config.gap:.1f} nm")
        print(f"  Trench gate x_length: {self.trench_gate.config.x_length:.1f} nm")
        print(f"  Trench gate y_width: {self.trench_gate.config.y_width:.1f} nm")


# --------------------------------------------------------------------
# 1V basis 계산
# --------------------------------------------------------------------

def compute_split_trench_basis(
    structure: SplitTrenchStructure,
    grid: ElectrostaticsGrid,
) -> Dict[str, np.ndarray]:
    """
    Split gate + Trench gate 각각의 1V 기준 전위 basis를 계산.

    파라미터:
        structure : SplitTrenchStructure
        grid      : ElectrostaticsGrid

    반환:
        {
            "split_left": φ_left^(1V),
            "split_right": φ_right^(1V),
            "trench": φ_trench^(1V),
        }
    """
    basis: Dict[str, np.ndarray] = {}

    # Split gate basis
    split_shapes = structure.split_gate.shapes
    if split_shapes is None:
        raise RuntimeError("Split gate shapes not built.")

    basis["split_left"] = _polygon_potential_basis_1V(split_shapes["left"], grid)
    basis["split_right"] = _polygon_potential_basis_1V(split_shapes["right"], grid)

    # Trench gate basis
    trench_shapes = structure.trench_gate.shapes
    if trench_shapes is None:
        raise RuntimeError("Trench gate shapes not built.")

    basis["trench"] = _polygon_potential_basis_1V(trench_shapes["trench"], grid)

    return basis


# --------------------------------------------------------------------
# 결합 전위 계산: 메인 함수
# --------------------------------------------------------------------

def compute_split_trench_potential(
    structure: SplitTrenchStructure,
    voltages: SplitTrenchVoltages,
    grid: ElectrostaticsGrid,
    screened: bool = False,
    label: str = "split_trench",
) -> PotentialMap:
    """
    Split gate + Trench gate 결합 전위 φ(x, y)를 계산.

    φ_total = V_left * φ_left^(1V) + V_right * φ_right^(1V) + V_trench * φ_trench^(1V)

    파라미터:
        structure : SplitTrenchStructure
        voltages  : SplitTrenchVoltages
        grid      : ElectrostaticsGrid
        screened  : True이면 screening 적용
        label     : PotentialMap에 붙일 이름

    반환:
        PotentialMap(phi=φ_total(x, y), label=label)
    """
    # 1V basis 계산
    basis = compute_split_trench_basis(structure, grid)

    # 전압 추출
    split_V = voltages.split_voltages.as_dict()
    trench_V = voltages.trench_voltages.as_dict()

    V_left = float(split_V["left"])
    V_right = float(split_V["right"])
    V_trench = float(trench_V["trench"])

    # 선형 결합
    phi_total = (
        V_left * basis["split_left"]
        + V_right * basis["split_right"]
        + V_trench * basis["trench"]
    )

    # (옵션) Screening 적용
    if screened:
        phi_total = apply_linear_screening(phi_total)

    return PotentialMap(phi=phi_total, label=label)


# --------------------------------------------------------------------
# 개별 gate 전위 계산 (디버깅/비교용)
# --------------------------------------------------------------------

def compute_individual_potentials(
    structure: SplitTrenchStructure,
    voltages: SplitTrenchVoltages,
    grid: ElectrostaticsGrid,
    screened: bool = False,
) -> Dict[str, PotentialMap]:
    """
    각 gate의 전위를 개별적으로 계산하여 반환.
    디버깅 및 기여도 분석용.

    반환:
        {
            "split": PotentialMap (split gate만의 전위),
            "trench": PotentialMap (trench gate만의 전위),
            "total": PotentialMap (결합 전위),
        }
    """
    basis = compute_split_trench_basis(structure, grid)

    split_V = voltages.split_voltages.as_dict()
    trench_V = voltages.trench_voltages.as_dict()

    V_left = float(split_V["left"])
    V_right = float(split_V["right"])
    V_trench = float(trench_V["trench"])

    # Split gate 전위
    phi_split = V_left * basis["split_left"] + V_right * basis["split_right"]
    if screened:
        phi_split = apply_linear_screening(phi_split)

    # Trench gate 전위
    phi_trench = V_trench * basis["trench"]
    if screened:
        phi_trench = apply_linear_screening(phi_trench)

    # 결합 전위
    phi_total = phi_split + phi_trench

    return {
        "split": PotentialMap(phi=phi_split, label="split_only"),
        "trench": PotentialMap(phi=phi_trench, label="trench_only"),
        "total": PotentialMap(phi=phi_total, label="split_trench_total"),
    }