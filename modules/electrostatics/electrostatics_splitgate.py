# modules/electrostatics/electrostatics_splitgate.py

"""
electrostatics_splitgate.py

Split gate 구조가 2DEG 평면에서 만들어내는 전위를
Davies 폴리곤 공식을 사용해 계산하는 모듈.

역할:
- GateStructure (splitgate geometry)
- SplitGateVoltages (왼쪽/오른쪽 gate 전압)
- ElectrostaticsGrid (x, y, depth_d)

를 입력받아, 2DEG 평면에서의 전위 φ(x, y)를 계산하여 PotentialMap으로 반환.

주의:
- 여기서는 "bare" electrostatic potential만 계산한다.
- screening(전자 스크리닝 효과)는 core.apply_linear_screening(...) 등으로
  나중에 별도로 처리 가능하도록 분리한다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Literal

import numpy as np

from modules.electrostatics.electrostatics_core import (
    ElectrostaticsGrid,
    PotentialMap,
    apply_linear_screening,
)

from modules.gate_structures.splitgate import GateStructure
from modules.gate_voltages.voltages_splitgate import SplitGateVoltages

Point = Tuple[float, float]
GateName = Literal["left", "right"]


# --------------------------------------------------------------------
# Davies polygon formula 기반: 단일 polygon gate의 1 V basis 계산
# --------------------------------------------------------------------

def _cot(x: np.ndarray) -> np.ndarray:
    """
    수치적 안정성을 고려한 cot(x) 계산.

    tan(x)=0 근처에서 분모가 0이 되는 것을 막기 위해 작은 값을 더해준다.
    """
    tan_x = np.tan(x)
    tan_x[tan_x == 0] += 1e-15
    return 1.0 / tan_x


def polygon_potential_basis_1V(
    vertices: List[Point],
    grid: ElectrostaticsGrid,
) -> np.ndarray:
    """
    Davies 폴리곤 공식을 사용하여,
    "주어진 polygon gate에 1 V가 인가되었을 때"
    2DEG 평면(grid.depth_d)에서의 전위 분포 φ(x, y)를 계산한다.

    파라미터:
        vertices : [(x1, y1), (x2, y2), ..., (xN, yN)] 형태의 polygon 꼭짓점 리스트.
                   시계/반시계 방향 순서는 크게 중요하지 않지만
                   닫힌 경로 형태여야 한다.
        grid     : ElectrostaticsGrid (x, y, depth_d)

    반환:
        phi_1V : 2D numpy 배열, shape = (Ny, Nx)
                 이 gate에 1 V를 인가했을 때의 전위 φ(x, y).
    """
    # grid 좌표
    X, Y = grid.meshgrid()  # shape: (Ny, Nx)
    z = grid.depth_d        # 2DEG 깊이

    # 복소수 plane에서 다루기 위한 r, p
    r = X + 1j * Y  # observation points
    pn = len(vertices)

    p = np.array(vertices, dtype=float)
    p_c = p[:, 0] + 1j * p[:, 1]  # 복소수 좌표로 변환

    basis = np.zeros_like(r, dtype=float)

    for m in range(pn):
        # cyclic indexing
        p0 = p_c[(m - 1) % pn]
        p1 = p_c[m]
        p2 = p_c[(m + 1) % pn]

        # Davies 식과 동일한 각도 a, b
        a = -np.angle((r - p1) / (p0 - p1))
        b =  np.angle((r - p1) / (p2 - p1))

        # sin(γ) = z / sqrt(|r - p1|^2 + z^2)
        dist_sq = np.abs(r - p1) ** 2
        sin_g = z / np.sqrt(dist_sq + z**2)

        # basis += [ (arctan(cot(a)) - arctan(sin_g * cot(a))) +
        #            (arctan(cot(b)) - arctan(sin_g * cot(b))) ]
        cot_a = _cot(a)
        cot_b = _cot(b)

        basis += (
            (np.arctan(cot_a) - np.arctan(sin_g * cot_a))
            + (np.arctan(cot_b) - np.arctan(sin_g * cot_b))
        )

    # 최종 1V 기준 전위 (기존 코드: self.basis = -basis/(2*pi))
    phi_1V = -basis / (2.0 * np.pi)
    return phi_1V


# --------------------------------------------------------------------
# Split gate 전위 계산: left/right polygon + voltages
# --------------------------------------------------------------------

def compute_splitgate_potential(
    gate_struct: GateStructure,
    gate_voltages: SplitGateVoltages,
    grid: ElectrostaticsGrid,
    screened: bool = False,
    label: str = "splitgate",
) -> PotentialMap:
    """
    SplitGate geometry + voltages를 사용해 2DEG 평면에서의 전위 φ(x, y)를 계산.

    파라미터:
        gate_struct : GateStructure (modules/gate_structures/splitgate.py)
                      내부에 최소한
                        - shapes: Dict["left" | "right", List[(x, y)]]
                      가 있다고 가정.
        gate_voltages : SplitGateVoltages
                        - V_left, V_right (Volt)
        grid          : ElectrostaticsGrid
        screened      : True이면 나중에 screening을 적용하도록 reserved.
                        현재는 apply_linear_screening stub를 호출.
        label         : PotentialMap에 붙일 이름.

    반환:
        PotentialMap(phi=φ_total(x, y), label=label)
    """
    # GateStructure가 가진 polygon 정보 추출
    # (이 부분은 GateStructure 구현에 따라 조정 필요)
    shapes: Dict[GateName, List[Point]] = gate_struct.shapes  # type: ignore[attr-defined]

    if "left" not in shapes or "right" not in shapes:
        raise KeyError("gate_struct.shapes must contain 'left' and 'right' polygons.")

    left_poly = shapes["left"]
    right_poly = shapes["right"]

    # 1V 기준 basis 계산
    phi_left_1V = polygon_potential_basis_1V(left_poly, grid)
    phi_right_1V = polygon_potential_basis_1V(right_poly, grid)

    # 현재 인가된 전압 (Volt 단위)
    V_dict = gate_voltages.as_dict()
    V_left = float(V_dict["left"])
    V_right = float(V_dict["right"])

    # 선형 결합: φ_total = V_left * φ_left^(1V) + V_right * φ_right^(1V)
    phi_total = V_left * phi_left_1V + V_right * phi_right_1V

    # (옵션) Screening 적용
    if screened:
        phi_total = apply_linear_screening(phi_total)

    return PotentialMap(phi=phi_total, label=label)
