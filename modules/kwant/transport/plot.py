# modules/kwant/plot.py

"""
plot.py

Kwant transport 계산 결과 시각화:
- Transmission T(E)
- Conductance G(E)
- Wavefunction |ψ|²
- Conductance plateaus
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm


def plot_transmission(
    result,  # TransportResult
    highlight_ef: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Transmission vs Energy",
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[Figure, Axes]:
    """
    Transmission T(E) 그래프.
    
    파라미터:
        result: TransportResult
        highlight_ef: Fermi 에너지 표시
        xlim: x축 범위 (meV)
        ylim: y축 범위
        title: 제목
        figsize: figure 크기
    
    반환:
        (fig, ax)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    energies_meV = result.energies * 1e3
    
    # Transmission
    ax.plot(energies_meV, result.transmission, 'b-', linewidth=2, label='T(E)')
    
    # Fermi energy 표시
    if highlight_ef:
        ef_meV = result.fermi_energy * 1e3
        ax.axvline(ef_meV, color='red', linestyle='--', linewidth=2, 
                  label=f'E_F = {ef_meV:.2f} meV')
        
        # T(E_F) 표시
        T_ef = np.interp(result.fermi_energy, result.energies, result.transmission)
        ax.plot(ef_meV, T_ef, 'ro', markersize=10, 
               label=f'T(E_F) = {T_ef:.3f}')
    
    # Integer transmission lines (conductance plateaus)
    for n in range(1, 6):
        ax.axhline(n, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, n, f'T={n}', 
               va='center', ha='right', fontsize=9, color='gray')
    
    ax.set_xlabel('Energy [meV]', fontsize=12)
    ax.set_ylabel('Transmission T(E)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    return fig, ax


def plot_conductance(
    result,  # TransportResult
    highlight_ef: bool = True,
    use_SI_units: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Conductance vs Energy",
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[Figure, Axes]:
    """
    Conductance G(E) 그래프.
    
    파라미터:
        result: TransportResult
        highlight_ef: Fermi 에너지 표시
        use_SI_units: SI 단위 사용 (기본: 2e²/h)
        xlim: x축 범위 (meV)
        ylim: y축 범위
        title: 제목
        figsize: figure 크기
    
    반환:
        (fig, ax)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    energies_meV = result.energies * 1e3
    
    if use_SI_units:
        conductance = result.conductance_SI * 1e6  # μS
        ylabel = 'Conductance [μS]'
        G0 = 77.48  # μS
    else:
        conductance = result.conductance
        ylabel = 'Conductance [2e²/h]'
        G0 = 1.0
    
    # Conductance
    ax.plot(energies_meV, conductance, 'b-', linewidth=2, label='G(E)')
    
    # Fermi energy 표시
    if highlight_ef:
        ef_meV = result.fermi_energy * 1e3
        ax.axvline(ef_meV, color='red', linestyle='--', linewidth=2, 
                  label=f'E_F = {ef_meV:.2f} meV')
        
        # G(E_F) 표시
        G_ef = np.interp(result.fermi_energy, result.energies, conductance)
        ax.plot(ef_meV, G_ef, 'ro', markersize=10, 
               label=f'G(E_F) = {G_ef:.3f}')
    
    # Quantization plateaus
    for n in range(1, 6):
        ax.axhline(n * G0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        if use_SI_units:
            ax.text(ax.get_xlim()[1] * 0.95, n * G0, f'{n}G₀', 
                   va='center', ha='right', fontsize=9, color='gray')
        else:
            ax.text(ax.get_xlim()[1] * 0.95, n * G0, f'{n}', 
                   va='center', ha='right', fontsize=9, color='gray')
    
    ax.set_xlabel('Energy [meV]', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    return fig, ax


def plot_wavefunction_2d(
    positions: np.ndarray,
    wf_density: np.ndarray,
    energy_meV: float,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'hot',
    use_log_scale: bool = False,
) -> Tuple[Figure, Axes]:
    """
    2D 파동함수 |ψ|² 시각화.
    
    파라미터:
        positions: 격자점 좌표 [nm], shape: (n_sites, 2)
        wf_density: |ψ|², shape: (n_sites,)
        energy_meV: 에너지 [meV]
        title: 제목
        figsize: figure 크기
        cmap: colormap
        use_log_scale: log scale 사용
    
    반환:
        (fig, ax)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    if title is None:
        title = f'Wavefunction |ψ|² at E = {energy_meV:.2f} meV'
    
    if use_log_scale:
        norm = LogNorm(vmin=wf_density[wf_density > 0].min(), 
                      vmax=wf_density.max())
    else:
        norm = None
    
    scatter = ax.scatter(x, y, c=wf_density, s=20, cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('|ψ|²', fontsize=11)
    
    ax.set_xlabel('x [nm]', fontsize=12)
    ax.set_ylabel('y [nm]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def plot_transmission_and_conductance(
    result,  # TransportResult
    highlight_ef: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "Transport Properties",
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Transmission과 Conductance를 나란히 표시.
    
    파라미터:
        result: TransportResult
        highlight_ef: Fermi 에너지 표시
        xlim: x축 범위 (meV)
        title: 제목
        figsize: figure 크기
    
    반환:
        (fig, (ax1, ax2))
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    energies_meV = result.energies * 1e3
    
    # Left: Transmission
    ax1.plot(energies_meV, result.transmission, 'b-', linewidth=2)
    
    if highlight_ef:
        ef_meV = result.fermi_energy * 1e3
        ax1.axvline(ef_meV, color='red', linestyle='--', linewidth=2)
        T_ef = np.interp(result.fermi_energy, result.energies, result.transmission)
        ax1.plot(ef_meV, T_ef, 'ro', markersize=8)
    
    for n in range(1, 6):
        ax1.axhline(n, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Energy [meV]', fontsize=11)
    ax1.set_ylabel('Transmission T(E)', fontsize=11)
    ax1.set_title('Transmission', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Right: Conductance
    ax2.plot(energies_meV, result.conductance, 'g-', linewidth=2)
    
    if highlight_ef:
        G_ef = np.interp(result.fermi_energy, result.energies, result.conductance)
        ax2.axvline(ef_meV, color='red', linestyle='--', linewidth=2, 
                   label=f'E_F = {ef_meV:.2f} meV')
        ax2.plot(ef_meV, G_ef, 'ro', markersize=8, 
                label=f'G = {G_ef:.3f} × 2e²/h')
    
    for n in range(1, 6):
        ax2.axhline(n, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Energy [meV]', fontsize=11)
    ax2.set_ylabel('Conductance [2e²/h]', fontsize=11)
    ax2.set_title('Conductance', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_multiple_wavefunctions(
    result,  # TransportResult with wf_data
    max_plots: int = 4,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    cmap: str = 'hot',
) -> Tuple[Figure, np.ndarray]:
    """
    여러 에너지에서의 파동함수를 grid로 표시.
    
    파라미터:
        result: TransportResult (wf_data 포함)
        max_plots: 최대 표시 개수
        figsize_per_plot: 각 subplot 크기
        cmap: colormap
    
    반환:
        (fig, axes)
    """
    
    if result.wf_data is None or len(result.wf_data) == 0:
        print("No wavefunction data available")
        return None, None
    
    n_plots = min(max_plots, len(result.wf_data))
    ncols = min(2, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_plots):
        positions, wf_density = result.wf_data[i]
        energy_meV = result.wf_energies[i] * 1e3
        
        ax = axes[i]
        
        x = positions[:, 0]
        y = positions[:, 1]
        
        scatter = ax.scatter(x, y, c=wf_density, s=15, cmap=cmap)
        
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xlabel('x [nm]', fontsize=10)
        ax.set_ylabel('y [nm]', fontsize=10)
        ax.set_title(f'E = {energy_meV:.2f} meV', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # 빈 subplot 숨기기
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle('Wavefunctions at Different Energies', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


def print_transport_analysis(
    result,  # TransportResult
    temperature_K: float = 0.1,
) -> None:
    """
    Transport 계산 결과 요약 출력.
    
    파라미터:
        result: TransportResult
        temperature_K: 측정 온도 [K]
    """
    
    print("\n" + "=" * 60)
    print("Kwant Transport Calculation Results")
    print("=" * 60)
    
    # Energy range
    print(f"\nEnergy range:")
    print(f"  E_min: {result.energies[0]*1e3:.3f} meV")
    print(f"  E_max: {result.energies[-1]*1e3:.3f} meV")
    print(f"  N_points: {len(result.energies)}")
    
    # Fermi energy
    ef_meV = result.fermi_energy * 1e3
    print(f"\nFermi energy:")
    print(f"  E_F: {ef_meV:.3f} meV")
    
    # Lead modes
    print(f"\nLead properties:")
    print(f"  Number of propagating modes at E_F: {result.n_modes_lead}")
    
    # Transmission and conductance at E_F
    T_ef = np.interp(result.fermi_energy, result.energies, result.transmission)
    G_ef = np.interp(result.fermi_energy, result.energies, result.conductance)
    G_ef_SI = np.interp(result.fermi_energy, result.energies, result.conductance_SI)
    
    print(f"\nTransport at E_F:")
    print(f"  T(E_F): {T_ef:.4f}")
    print(f"  G(E_F): {G_ef:.4f} × 2e²/h")
    print(f"          {G_ef_SI*1e6:.2f} μS")
    
    # Conductance quantization
    k_B_eV_K = 8.617333262e-5
    thermal_energy = k_B_eV_K * temperature_K
    
    print(f"\nConductance quantization analysis (T = {temperature_K} K):")
    print(f"  Thermal energy k_B T: {thermal_energy*1e3:.3f} meV")
    
    # Plateau 분석
    G_tolerance = 0.1  # 2e²/h 단위
    for n in range(1, 6):
        in_plateau = np.abs(result.conductance - n) < G_tolerance
        if np.any(in_plateau):
            E_plateau = result.energies[in_plateau]
            print(f"  Plateau at G = {n} × 2e²/h:")
            print(f"    Energy range: [{E_plateau.min()*1e3:.2f}, {E_plateau.max()*1e3:.2f}] meV")
            print(f"    Width: {(E_plateau.max() - E_plateau.min())*1e3:.2f} meV")
    
    # Wavefunction info
    if result.wf_data is not None:
        print(f"\nWavefunction calculations:")
        print(f"  Number of energies: {len(result.wf_energies)}")
        for E in result.wf_energies:
            print(f"    E = {E*1e3:.2f} meV")
    
    print("=" * 60 + "\n")