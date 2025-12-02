# modules/electrostatics/electrostatics_core.py

"""
electrostatics_core.py

Electrostatics 모듈들이 공통으로 사용하는
- 2DEG 평면 grid 정의
- 전위 맵 래퍼
- 기본 유틸리티 함수

를 제공하는 코어 모듈.

이 파일은 gate 종류(splitgate, topgate 등)에 대해 아무것도 모릅니다.
단지 "x, y, depth"와 "phi(x, y)"라는 데이터 구조만 정의합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ElectrostaticsGrid:
    """
    2DEG 평면에서 전위를 계산할 grid 정의.

    x: 1D array, x 좌표 (nm 등)
    y: 1D array, y 좌표 (nm 등)
    depth_d: 2DEG 깊이 d (nm) - 표면 z=0 기준으로 z=+d 위치에 2DEG가 있다고 가정

    주의:
        meshgrid를 만들 때는 항상 indexing="xy"로 사용합니다.
    """
    x: np.ndarray
    y: np.ndarray
    depth_d: float

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D meshgrid (X, Y)를 반환.
        shape는 (Ny, Nx) 이며,
        X[i, j] = x[j], Y[i, j] = y[i] 형태 (indexing="xy").
        """
        return np.meshgrid(self.x, self.y, indexing="xy")

    @property
    def shape(self) -> Tuple[int, int]:
        """(Ny, Nx) 형태의 grid shape를 반환."""
        return (self.y.size, self.x.size)


@dataclass
class PotentialMap:
    """
    2DEG 평면에서의 전위 φ(x, y)를 담는 래퍼.

    phi : 2D 배열, shape = (Ny, Nx)
    label : 이 전위 맵을 구분하기 위한 이름 (예: "splitgate", "topgate1", "total" 등)
    """
    phi: np.ndarray
    label: str = "unnamed"

    def copy(self, label: str | None = None) -> "PotentialMap":
        """전위 데이터를 복사하여 새 PotentialMap을 만든다."""
        new_label = self.label if label is None else label
        return PotentialMap(phi=self.phi.copy(), label=new_label)


def make_uniform_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
    depth_d: float,
) -> ElectrostaticsGrid:
    """
    직사각형 영역 [x_min, x_max] × [y_min, y_max] 에 대해
    균일한 spacing을 가진 ElectrostaticsGrid를 생성한다.

    파라미터:
        x_min, x_max : x 좌표 범위
        y_min, y_max : y 좌표 범위
        nx, ny       : x, y 방향 grid 포인트 개수
        depth_d      : 2DEG 깊이 (nm)

    반환:
        ElectrostaticsGrid
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    return ElectrostaticsGrid(x=x, y=y, depth_d=depth_d)


# ---- (옵션) 나중에 screening 등을 구현할 때 쓸 수 있는 stub ---- #

def apply_linear_screening(
    phi_bare: np.ndarray,
    *args,
    **kwargs,
) -> np.ndarray:
    """
    TODO: Davies 논문 기반 linear-response screening을 적용하는 함수.

    현재는 stub으로서 bare potential을 그대로 반환한다.
    나중에:
        φ_scr(r, d) ∝ ∂φ/∂z (z=d) 등을 사용해 구현 가능.
    """
    # 아직 구현 전: 일단 그대로 반환
    return phi_bare
