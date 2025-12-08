# modules/quantum/schrodinger_1d.py

"""
schrodinger_1d.py

1D Time-Independent Schrödinger Equation Solver

Solves:
    [-ℏ²/(2m*) d²/dy² + V(y)] ψ(y) = E ψ(y)

Uses finite difference method to convert PDE to eigenvalue problem.
"""

import numpy as np
from scipy import linalg
from typing import Tuple
import matplotlib.pyplot as plt


def solve_schrodinger_1d(
    x: np.ndarray,
    V: np.ndarray,
    m_eff_kg: float,
    n_states: int = 10,
    return_all: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D 슈뢰딩거 방정식을 유한 차분법으로 풀기.
    
    파라미터:
        x: 위치 배열 [nm], shape: (N,)
        V: 전위 에너지 [eV], shape: (N,)
        m_eff_kg: 유효 질량 [kg]
        n_states: 계산할 상태 개수
        return_all: True면 모든 계산 정보 반환
    
    반환:
        energies: 에너지 고유값 [eV], shape: (n_states,)
        wavefunctions: 파동함수, shape: (n_states, N)
    """
    
    # 단위 변환 상수
    HBAR_J_S = 1.054571817e-34  # ℏ [J·s]
    E_CHARGE_C = 1.602176634e-19  # e [C]
    NM_TO_M = 1e-9
    
    # Grid 파라미터
    N = len(x)
    dx_nm = x[1] - x[0]  # nm
    dx_m = dx_nm * NM_TO_M  # m
    
    # Kinetic energy coefficient
    # t = ℏ² / (2 m* dx²) [J]
    t_J = (HBAR_J_S**2) / (2.0 * m_eff_kg * dx_m**2)
    
    # Convert to eV
    t = t_J / E_CHARGE_C  # [eV]
    
    # Hamiltonian 행렬 구성
    # H_ij = T_ij + V_i δ_ij
    
    # Kinetic part: tridiagonal
    # T = t × [-2  1  0  0 ...]
    #         [ 1 -2  1  0 ...]
    #         [ 0  1 -2  1 ...]
    #         ...
    
    diag_main = np.ones(N) * 2.0 * t  # Main diagonal: 2t
    diag_off = np.ones(N-1) * (-t)    # Off-diagonals: -t
    
    H = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)
    
    # Potential part: diagonal
    H += np.diag(V)
    
    # 경계조건: ψ(x_0) = ψ(x_N-1) = 0
    # 이미 포함됨 (finite grid → natural boundary)
    
    # Eigenvalue problem 풀기
    # H ψ = E ψ
    eigenvalues, eigenvectors = linalg.eigh(H)
    
    # 처음 n_states개만 선택
    energies = eigenvalues[:n_states]
    wavefunctions = eigenvectors[:, :n_states].T  # shape: (n_states, N)
    
    # 파동함수 정규화 (이미 eigh가 해주지만 명시적으로)
    for i in range(n_states):
        norm = np.sqrt(np.trapezoid(wavefunctions[i]**2, x))
        wavefunctions[i] /= norm
    
    if return_all:
        return energies, wavefunctions, H, t, dx_nm
    else:
        return energies, wavefunctions


def plot_potential_and_levels(
    x: np.ndarray,
    V: np.ndarray,
    energies: np.ndarray,
    wavefunctions: np.ndarray = None,
    E_fermi: float = None,
    title: str = "1D Schrödinger: Potential and Subband Levels",
    figsize: Tuple[float, float] = (12, 8),
    show_wf: bool = True,
    wf_scale: float = 5.0,
    n_levels: int = 5,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Potential과 subband energy level을 시각화.
    
    파라미터:
        x: 위치 [nm]
        V: 전위 [eV]
        energies: 에너지 준위 [eV]
        wavefunctions: 파동함수 (optional)
        E_fermi: Fermi 에너지 [eV] (optional)
        title: 그래프 제목
        figsize: Figure 크기
        show_wf: 파동함수 표시 여부
        wf_scale: 파동함수 스케일 (meV 단위)
        n_levels: 표시할 레벨 개수
    
    반환:
        (fig, ax)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # eV → meV 변환
    V_meV = V * 1e3
    energies_meV = energies * 1e3
    
    # Potential curve
    ax.plot(x, V_meV, 'b-', linewidth=2.5, label='Potential V(y)', zorder=1)
    
    # Subband energy levels
    n_to_show = min(n_levels, len(energies))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_to_show))
    
    for i in range(n_to_show):
        E_meV = energies_meV[i]
        
        # Horizontal line for energy level
        ax.axhline(E_meV, color=colors[i], linestyle='--', linewidth=2, 
                  alpha=0.7, zorder=2)
        
        # Label
        ax.text(x[-1] * 0.95, E_meV, f'$E_{i}$ = {E_meV:.2f} meV', 
               va='center', ha='right', fontsize=10, 
               color=colors[i], fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=colors[i], alpha=0.8))
        
        # Wavefunction (if provided)
        if show_wf and wavefunctions is not None:
            psi = wavefunctions[i]
            psi_squared = psi**2
            
            # Scale to make visible (arbitrary units)
            psi_scaled = psi_squared / np.max(np.abs(psi_squared)) * wf_scale
            
            # Shift to energy level
            psi_plot = psi_scaled + E_meV
            
            ax.fill_between(x, E_meV, psi_plot, 
                           color=colors[i], alpha=0.3, zorder=3)
            ax.plot(x, psi_plot, color=colors[i], linewidth=1.5, 
                   alpha=0.8, zorder=4)
    
    # Fermi energy line
    if E_fermi is not None:
        E_f_meV = E_fermi * 1e3
        ax.axhline(E_f_meV, color='red', linestyle='-', linewidth=3, 
                  alpha=0.8, label=f'$E_F$ = {E_f_meV:.2f} meV', zorder=5)
        
        # Count propagating modes
        n_modes = np.sum(energies < E_fermi)
        ax.text(x[0] * 0.95, E_f_meV, f'  {n_modes} modes', 
               va='bottom', ha='left', fontsize=11, color='red',
               fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Position y [nm]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy [meV]', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Y-axis range
    y_min = min(np.min(V_meV), energies_meV[0]) - 10
    if E_fermi is not None:
        y_max = max(E_f_meV, energies_meV[min(n_to_show-1, len(energies)-1)]) + 20
    else:
        y_max = energies_meV[min(n_to_show-1, len(energies)-1)] + 20
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    return fig, ax


def print_subband_analysis(
    energies: np.ndarray,
    E_fermi: float = None,
    verbose: bool = True,
) -> dict:
    """
    Subband 에너지 분석 결과 출력 및 반환.
    
    파라미터:
        energies: 에너지 준위 [eV]
        E_fermi: Fermi 에너지 [eV] (optional)
        verbose: 출력 여부
    
    반환:
        분석 결과 dictionary
    """
    
    energies_meV = energies * 1e3
    
    # Subband spacings
    spacings_meV = np.diff(energies_meV)
    avg_spacing_meV = np.mean(spacings_meV[:min(3, len(spacings_meV))])
    
    # Room temperature requirement
    kT_300K_meV = 25.9  # k_B × 300K in meV
    RT_threshold_meV = 5 * kT_300K_meV  # 129.5 meV
    
    if verbose:
        print("\n" + "="*60)
        print("1D Schrödinger Equation: Subband Analysis")
        print("="*60)
        
        print(f"\nSubband energies (first {min(5, len(energies))} states):")
        for i, E in enumerate(energies[:5]):
            print(f"  E_{i} = {E*1e3:8.3f} meV")
        
        print(f"\nSubband spacings:")
        for i, dE in enumerate(spacings_meV[:4]):
            print(f"  ΔE_{i}{i+1} = E_{i+1} - E_{i} = {dE:.3f} meV")
        
        print(f"\nAverage spacing (first 3): {avg_spacing_meV:.3f} meV")
        
        if E_fermi is not None:
            E_f_meV = E_fermi * 1e3
            n_modes = np.sum(energies < E_fermi)
            print(f"\nFermi energy: E_F = {E_f_meV:.3f} meV")
            print(f"Propagating modes at E_F: {n_modes}")
            print(f"Expected conductance: G = {n_modes} × (2e²/h)")
        
        print(f"\n{'─'*60}")
        print("Room temperature quantization assessment:")
        print(f"  Thermal energy at 300K: k_B T = {kT_300K_meV:.2f} meV")
        print(f"  Required spacing: ΔE > 5 k_B T = {RT_threshold_meV:.1f} meV")
        print(f"  Current spacing: ΔE = {avg_spacing_meV:.3f} meV")
        
        if avg_spacing_meV > RT_threshold_meV:
            print(f"  ✓ Room-temperature quantization possible!")
        else:
            ratio = RT_threshold_meV / avg_spacing_meV
            print(f"  ✗ Need {ratio:.1f}× larger spacing for room temperature")
            print(f"  → Suggestions:")
            print(f"     • Reduce gap width")
            print(f"     • Increase gate voltage magnitude")
            print(f"     • Optimize trench voltage")
        
        print("="*60 + "\n")
    
    # Return results
    return {
        'energies_meV': energies_meV,
        'spacings_meV': spacings_meV,
        'average_spacing_meV': avg_spacing_meV,
        'n_modes': np.sum(energies < E_fermi) if E_fermi is not None else None,
        'RT_possible': avg_spacing_meV > RT_threshold_meV,
        'RT_ratio': avg_spacing_meV / RT_threshold_meV,
    }


def create_harmonic_potential(
    x: np.ndarray,
    V0: float = 0.0,
    omega_meV: float = 5.0,
) -> np.ndarray:
    """
    테스트용 조화진동자 potential 생성.
    
    V(y) = V0 + (1/2) m* ω² y²
    
    파라미터:
        x: 위치 [nm]
        V0: Minimum energy [eV]
        omega_meV: ℏω [meV]
    
    반환:
        V: Potential [eV]
    """
    
    # Convert to SI
    x_m = x * 1e-9
    omega_eV = omega_meV * 1e-3
    
    # For parabola: V = V0 + (1/2) k y²
    # where k is chosen to give desired ℏω
    # ℏω = √(k/m*) × ℏ
    # → k = (ℏω)² / ℏ² × m*
    
    # Simplification: just use parabola with scale
    # V(y) = V0 + α y²
    # Choose α to give reasonable energy scale
    
    alpha = omega_eV / (2.0 * (x_m[-1]**2))  # Scale factor
    
    V = V0 + alpha * x_m**2
    
    return V


# Example usage / Test
if __name__ == "__main__":
    
    print("="*70)
    print("1D Schrödinger Equation Solver - Test")
    print("="*70)
    
    # Physical parameters (GaAs)
    M_EFF_GAAS_KG = 0.067 * 9.1093837015e-31  # 0.067 m_e
    
    # Grid
    y = np.linspace(-50, 50, 501)  # nm
    
    # Test potential: Harmonic oscillator
    print("\nTest 1: Harmonic Oscillator")
    V_harmonic = create_harmonic_potential(y, V0=-0.010, omega_meV=5.0)
    
    # Solve
    energies, wavefunctions = solve_schrodinger_1d(
        x=y,
        V=V_harmonic,
        m_eff_kg=M_EFF_GAAS_KG,
        n_states=5,
    )
    
    # Analyze
    E_fermi = 0.012  # 12 meV
    analysis = print_subband_analysis(energies, E_fermi=E_fermi)
    
    # Plot
    fig, ax = plot_potential_and_levels(
        x=y,
        V=V_harmonic,
        energies=energies,
        wavefunctions=wavefunctions,
        E_fermi=E_fermi,
        title="Test: Harmonic Oscillator Potential",
        show_wf=True,
        wf_scale=3.0,
        n_levels=5,
    )
    
    plt.savefig('/mnt/user-data/outputs/test_harmonic_oscillator.png', 
                dpi=150, bbox_inches='tight')
    print("Saved: test_harmonic_oscillator.png")
    
    # Test potential: Gaussian well
    print("\n" + "="*70)
    print("Test 2: Gaussian Well")
    V0 = -0.020  # -20 meV
    sigma = 15.0  # nm
    V_gaussian = V0 * np.exp(-(y**2) / (2*sigma**2))
    
    energies2, wavefunctions2 = solve_schrodinger_1d(
        x=y,
        V=V_gaussian,
        m_eff_kg=M_EFF_GAAS_KG,
        n_states=5,
    )
    
    analysis2 = print_subband_analysis(energies2, E_fermi=E_fermi)
    
    fig2, ax2 = plot_potential_and_levels(
        x=y,
        V=V_gaussian,
        energies=energies2,
        wavefunctions=wavefunctions2,
        E_fermi=E_fermi,
        title="Test: Gaussian Well Potential",
        show_wf=True,
        wf_scale=4.0,
        n_levels=5,
    )
    
    plt.savefig('/mnt/user-data/outputs/test_gaussian_well.png', 
                dpi=150, bbox_inches='tight')
    print("Saved: test_gaussian_well.png")
    
    print("\n" + "="*70)
    print("Tests completed! Check output directory for plots.")
    print("="*70)
    
    plt.show()