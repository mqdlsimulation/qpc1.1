# modules/constants/physics.py

"""
physics.py

시뮬레이션에서 공통으로 사용하는 물리 상수 및
2DEG(AlGaAs/GaAs) 기본 파라미터를 정의한 모듈.

단위:
    - 에너지: eV
    - 길이   : nm
    - 전자 밀도: cm^-2
    - 전도도: cm^2/(V·s)
"""

from __future__ import annotations

import numpy as np

# -----------------------
# 기본 물리 상수
# -----------------------

# 전자 기본 전하 [C]
E_CHARGE_C: float = 1.602176634e-19

# 플랑크 상수 (reduced Planck) [J s]
HBAR_J_S: float = 1.054571817e-34

# 플랑크 상수 (reduced Planck) [eV s]
HBAR_EV_S: float = HBAR_J_S / E_CHARGE_C  # ≈ 6.582119569e-16 eV·s

# 전자 질량 [kg]
M_ELECTRON_KG: float = 9.1093837015e-31

# 볼츠만 상수 [eV/K]
KB_EV_K: float = 8.617333262e-5

# -----------------------
# 재료 상수 (GaAs 등)
# -----------------------

# GaAs 유효 질량 m*/m0
M_EFF_GAAS_MASS: float = 0.067
M_EFF_GAAS_KG: float = M_EFF_GAAS_MASS * M_ELECTRON_KG

# GaAs 유전상수
EPSILON_R_GAAS: float = 12.9

# -----------------------
# 2DEG 기본 파라미터 (실험값)
# -----------------------

# 전자 밀도 [cm^-2]
N_S_2DEG_CM2: float = 2.26e11

# 전자 밀도 [nm^-2] (단위 변환: 1 cm = 10^7 nm)
N_S_2DEG_NM2: float = N_S_2DEG_CM2 * 1e-14

# 이동도 (mobility) [cm^2/(V·s)]
MOBILITY_2DEG_CM2_VS: float = 4.1e6

# Fermi 에너지 계산 [eV]
# E_F = (π ℏ^2 n_s) / m*
# n_s [nm^-2], ℏ [eV·s], m* [kg]
EF_2DEG_EV: float = (
    np.pi * (HBAR_EV_S * 1e9)**2 * N_S_2DEG_NM2 / M_EFF_GAAS_KG
) / E_CHARGE_C  # [eV]

# Fermi 파장 λ_F [nm]
# λ_F = √(2π/n_s)
LAMBDA_F_2DEG_NM: float = np.sqrt(2.0 * np.pi / N_S_2DEG_NM2)

# Fermi 속도 v_F [m/s]
# v_F = √(2 E_F / m*)
VF_2DEG_MS: float = np.sqrt(
    2.0 * EF_2DEG_EV * E_CHARGE_C / M_EFF_GAAS_KG
)

# 평균 자유 경로 [nm]
# l_mean = v_F * m* * μ / e
L_MEAN_2DEG_NM: float = (
    VF_2DEG_MS * M_EFF_GAAS_KG * MOBILITY_2DEG_CM2_VS * 1e-4 / E_CHARGE_C
) * 1e9


# -----------------------
# 유틸리티 함수
# -----------------------

def hopping_from_ef_lambda(
    a_nm: float,
    ef_ev: float = EF_2DEG_EV,
    lambda_f_nm: float = LAMBDA_F_2DEG_NM,
) -> float:
    """
    연속계 파라미터 (E_F, λ_F)를
    2D tight-binding 모형의 hopping t[eV]로 매핑하는 함수.

    Davies / 기존 mesa 코드에서 사용한 관계식:
        t = E_F * λ_F^2 / (4 π^2 a^2)

    파라미터:
        a_nm        : 격자 간격 (mesh spacing) [nm]
        ef_ev       : Fermi 에너지 [eV]
        lambda_f_nm : Fermi 파장 [nm]

    반환:
        t [eV]
    """
    return ef_ev * (lambda_f_nm**2) / (4.0 * np.pi**2 * a_nm**2)


def print_2deg_parameters() -> None:
    """2DEG 파라미터 출력 (디버깅용)"""
    print("=" * 60)
    print("2DEG Parameters (AlGaAs/GaAs)")
    print("=" * 60)
    print(f"Electron density:      {N_S_2DEG_CM2:.2e} cm^-2")
    print(f"                       {N_S_2DEG_NM2:.4e} nm^-2")
    print(f"Mobility:              {MOBILITY_2DEG_CM2_VS:.2e} cm^2/(V·s)")
    print(f"Effective mass:        {M_EFF_GAAS_MASS:.3f} m₀")
    print(f"Fermi energy:          {EF_2DEG_EV*1e3:.3f} meV")
    print(f"Fermi wavelength:      {LAMBDA_F_2DEG_NM:.2f} nm")
    print(f"Fermi velocity:        {VF_2DEG_MS*1e-5:.2f} × 10^5 m/s")
    print(f"Mean free path:        {L_MEAN_2DEG_NM:.1f} nm")
    print("=" * 60)


if __name__ == "__main__":
    print_2deg_parameters()