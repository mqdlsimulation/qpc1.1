# simulations/qpc/simulation_voltage_sweep.py

"""
simulation_voltage_sweep.py

Split gate voltage를 sweep하며 각 voltage에서 transport 계산.
G(V_gate, E) 2D map 생성 및 최적 조건 탐색.

목적:
1. Pinch-off voltage 결정
2. 최적 quantization 조건 찾기
3. Subband evolution 관찰
4. 실험 데이터와 비교
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime

from modules.gate_structures.splitgate import build_split_gate_structure
from modules.gate_structures.trenchgate import build_trench_gate_structure
from modules.gate_voltages.voltages_splitgate import create_splitgate_voltages_from_mV
from modules.gate_voltages.voltages_trenchgate import create_trenchgate_voltages_from_mV
from modules.electrostatics.electrostatics_core import make_uniform_grid
from modules.electrostatics.electrostatics_split_trench import (
    SplitTrenchStructure,
    SplitTrenchVoltages,
    compute_split_trench_potential,
)
from modules.kwant.transport.transport import (
    KwantSystemConfig,
    run_transport_calculation,
    estimate_lattice_constant,
)
from modules.constants.physics import (
    M_EFF_GAAS_KG,
    EF_2DEG_EV,
    LAMBDA_F_2DEG_NM,
    hopping_from_ef_lambda,
    print_2deg_parameters,
)


def run_single_voltage(
    V_split_mV: float,
    V_trench_mV: float,
    gap: float,
    depth_d: float,
    structure: SplitTrenchStructure,
    grid,
    kwant_config: KwantSystemConfig,
    wf_energies_eV = None,
    verbose: bool = False,
):
    """
    단일 gate voltage에서 transport 계산.
    
    파라미터:
        V_split_mV: Split gate voltage [mV]
        V_trench_mV: Trench gate voltage [mV]
        gap: Split gate gap [nm]
        depth_d: 2DEG depth [nm]
        structure: SplitTrenchStructure
        grid: ElectrostaticsGrid
        kwant_config: KwantSystemConfig
        wf_energies_eV: Wavefunction 계산 에너지 (optional)
        verbose: 진행상황 출력
    
    반환:
        (result, phi_2d)
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"V_split = {V_split_mV:.1f} mV, V_trench = {V_trench_mV:.1f} mV")
        print(f"{'='*60}")
    
    # Voltage 설정
    split_volts = create_splitgate_voltages_from_mV(
        mode="symmetric",
        symmetric_Vg_mV=V_split_mV,
        V_left_mV=0.0,
        V_right_mV=0.0,
    )
    
    trench_volts = create_trenchgate_voltages_from_mV(V_trench_mV)
    
    voltages = SplitTrenchVoltages(
        split_voltages=split_volts,
        trench_voltages=trench_volts,
    )
    
    # Electrostatic potential 계산
    pot_map = compute_split_trench_potential(
        structure=structure,
        voltages=voltages,
        grid=grid,
        screened=False,
    )
    
    phi_2d = pot_map.phi
    
    if verbose:
        print(f"Potential range: [{np.min(phi_2d)*1e3:.1f}, {np.max(phi_2d)*1e3:.1f}] meV")
    
    # Grid 좌표
    grid_x = grid.x
    grid_y = grid.y
    
    # Transport 계산
    result = run_transport_calculation(
        config=kwant_config,
        potential_2d=phi_2d,
        grid_x=grid_x,
        grid_y=grid_y,
        fermi_energy=EF_2DEG_EV,
        compute_wf_at_energies=wf_energies_eV,
        verbose=verbose,
    )
    
    return result, phi_2d


def voltage_sweep_analysis(
    V_split_range: tuple,
    n_voltages: int,
    V_trench_mV: float = 1700.0,
    gap: float = 80.0,
    depth_d: float = 60.0,
    L_x_kwant: float = 80.0,
    L_y_kwant: float = 80.0,
    W_lead: float = 60.0,
    points_per_lambda: int = 8,
    n_energies: int = 100,
    save_results: bool = True,
):
    """
    Split gate voltage sweep 분석.
    
    파라미터:
        V_split_range: (V_min, V_max) [mV]
        n_voltages: Voltage 개수
        V_trench_mV: Trench gate voltage [mV]
        gap: Split gate gap [nm]
        depth_d: 2DEG depth [nm]
        L_x_kwant: Scattering region x 길이 [nm]
        L_y_kwant: Scattering region y 폭 [nm]
        W_lead: Lead 폭 [nm]
        points_per_lambda: 격자 해상도
        n_energies: 에너지 포인트 개수
        save_results: 결과 저장 여부
    
    반환:
        results_dict: {V_split: result}
    """
    
    print("=" * 70)
    print("Split Gate Voltage Sweep Analysis")
    print("=" * 70)
    
    print_2deg_parameters()
    
    # Voltage 범위
    V_split_array = np.linspace(V_split_range[0], V_split_range[1], n_voltages)
    
    print(f"\nVoltage sweep parameters:")
    print(f"  V_split range: [{V_split_range[0]:.0f}, {V_split_range[1]:.0f}] mV")
    print(f"  Number of points: {n_voltages}")
    print(f"  V_trench (fixed): {V_trench_mV:.0f} mV")
    
    # Gate 구조 생성 (한 번만)
    print(f"\nBuilding gate structures...")
    split_gate = build_split_gate_structure(
        gap=gap,
        gate_width_x=500.0,
        gate_length_y=80.0,
        two_deg_depth=depth_d,
        use_dut_offset=False,
        do_describe=False,
        do_plot=False,
    )
    
    trench_gate = build_trench_gate_structure(
        x_length=78.0,
        y_width=500.0,
        x_offset=0.0,
        y_offset=0.0,
        two_deg_depth=depth_d,
        split_shapes=split_gate.shapes,
        do_describe=False,
        do_plot=False,
        do_overlap_check=True,
    )
    
    structure = SplitTrenchStructure(
        split_gate=split_gate,
        trench_gate=trench_gate,
    )
    
    # Grid 생성 (한 번만)
    print(f"Creating electrostatics grid...")
    grid = make_uniform_grid(
        x_min=-300.0,
        x_max=300.0,
        y_min=-300.0,
        y_max=300.0,
        nx=501,
        ny=501,
        depth_d=depth_d,
    )
    
    # Kwant 설정 (한 번만)
    print(f"Configuring Kwant system...")
    a = estimate_lattice_constant(
        lambda_f_nm=LAMBDA_F_2DEG_NM,
        ef_ev=EF_2DEG_EV,
        target_points_per_wavelength=points_per_lambda,
    )
    
    t = hopping_from_ef_lambda(a, EF_2DEG_EV, LAMBDA_F_2DEG_NM)
    print(f"  Hopping energy t = {t*1e3:.3f} meV")
    
    kwant_config = KwantSystemConfig(
        a=a,
        L_x=L_x_kwant,
        L_y=L_y_kwant,
        W_lead=W_lead,
        t=t,
        m_eff_kg=M_EFF_GAAS_KG,
        E_min=0.0,
        E_max=EF_2DEG_EV * 1.5,
        n_energies=n_energies,
    )
    
    print(f"\n  Scattering region: {L_x_kwant} × {L_y_kwant} nm")
    print(f"  Centered at: x=0 (gap center)")
    print(f"  Lead width: {W_lead} nm")
    
    # Voltage sweep
    print(f"\n{'='*70}")
    print(f"Starting voltage sweep ({n_voltages} points)...")
    print(f"{'='*70}\n")
    
    results_dict = {}
    potentials_dict = {}
    
    for i, V_split in enumerate(V_split_array):
        print(f"[{i+1}/{n_voltages}] V_split = {V_split:.1f} mV")
        
        try:
            result, phi_2d = run_single_voltage(
                V_split_mV=V_split,
                V_trench_mV=V_trench_mV,
                gap=gap,
                depth_d=depth_d,
                structure=structure,
                grid=grid,
                kwant_config=kwant_config,
                verbose=False,
            )
            
            results_dict[V_split] = result
            potentials_dict[V_split] = phi_2d
            
            # 간단한 정보 출력
            G_ef = np.interp(EF_2DEG_EV, result.energies, result.conductance)
            print(f"  → G(E_F) = {G_ef:.3f} × 2e²/h")
            
        except Exception as e:
            print(f"  Error: {e}")
            results_dict[V_split] = None
            potentials_dict[V_split] = None
    
    print(f"\n{'='*70}")
    print("Voltage sweep completed!")
    print(f"{'='*70}\n")
    
    # 결과 저장
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/voltage_sweep")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"sweep_{timestamp}.pkl"
        
        save_data = {
            'V_split_array': V_split_array,
            'V_trench_mV': V_trench_mV,
            'results': results_dict,
            'potentials': potentials_dict,
            'config': {
                'gap': gap,
                'depth_d': depth_d,
                'L_x_kwant': L_x_kwant,
                'L_y_kwant': L_y_kwant,
                'W_lead': W_lead,
                'n_energies': n_energies,
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Results saved to: {filename}")
    
    return results_dict, potentials_dict, V_split_array


def plot_voltage_sweep_results(
    results_dict: dict,
    V_split_array: np.ndarray,
    fermi_energy: float = EF_2DEG_EV,
):
    """
    Voltage sweep 결과 시각화.
    """
    
    # G(E_F) vs V_gate
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    G_ef_array = []
    for V_split in V_split_array:
        result = results_dict[V_split]
        if result is not None:
            G_ef = np.interp(fermi_energy, result.energies, result.conductance)
            G_ef_array.append(G_ef)
        else:
            G_ef_array.append(0.0)
    
    G_ef_array = np.array(G_ef_array)
    
    ax1.plot(V_split_array, G_ef_array, 'b-o', linewidth=2, markersize=6)
    
    # Integer lines
    for n in range(1, 6):
        ax1.axhline(n, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.text(ax1.get_xlim()[1] * 0.95, n, f'{n}', 
                va='center', ha='right', fontsize=9, color='gray')
    
    ax1.set_xlabel('V_split [mV]', fontsize=12)
    ax1.set_ylabel('Conductance at E_F [2e²/h]', fontsize=12)
    ax1.set_title('Conductance vs Gate Voltage', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    plt.tight_layout()
    
    # G(E, V_gate) 2D map
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # 2D array 생성
    n_voltages = len(V_split_array)
    result_first = results_dict[V_split_array[0]]
    energies = result_first.energies * 1e3  # meV
    n_energies = len(energies)
    
    G_map = np.zeros((n_voltages, n_energies))
    
    for i, V_split in enumerate(V_split_array):
        result = results_dict[V_split]
        if result is not None:
            G_map[i, :] = result.conductance
    
    im = ax2.imshow(
        G_map,
        origin='lower',
        extent=[energies[0], energies[-1], V_split_array[0], V_split_array[-1]],
        aspect='auto',
        cmap='viridis',
        vmin=0,
        vmax=5,
    )
    
    # E_F line
    ef_meV = fermi_energy * 1e3
    ax2.axvline(ef_meV, color='red', linestyle='--', linewidth=2, label='E_F')
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Conductance [2e²/h]', fontsize=11)
    
    ax2.set_xlabel('Energy [meV]', fontsize=12)
    ax2.set_ylabel('V_split [mV]', fontsize=12)
    ax2.set_title('G(E, V_gate) 2D Map', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    
    # Selected voltage curves
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # 몇 개 voltage 선택
    n_select = min(5, len(V_split_array))
    indices = np.linspace(0, len(V_split_array)-1, n_select, dtype=int)
    
    for idx in indices:
        V_split = V_split_array[idx]
        result = results_dict[V_split]
        if result is not None:
            ax3.plot(result.energies * 1e3, result.conductance, 
                    linewidth=2, label=f'V = {V_split:.0f} mV')
    
    # E_F line
    ax3.axvline(ef_meV, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Integer lines
    for n in range(1, 6):
        ax3.axhline(n, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax3.set_xlabel('Energy [meV]', fontsize=12)
    ax3.set_ylabel('Conductance [2e²/h]', fontsize=12)
    ax3.set_title('G(E) at Selected Gate Voltages', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    plt.tight_layout()
    
    return fig1, fig2, fig3


def main():
    """메인 실행 함수"""
    
    # Voltage sweep 설정
    results, potentials, V_array = voltage_sweep_analysis(
        V_split_range=(-400, 0),  # mV
        n_voltages=51,  # points
        V_trench_mV=0,
        gap=80.0,
        depth_d=60.0,
        L_x_kwant=80.0,  # Gap 중심 포함 (x ∈ [-40, 40] nm)
        L_y_kwant=80.0,
        W_lead=60.0,
        points_per_lambda=8,
        n_energies=100,
        save_results=True,
    )
    
    # 결과 시각화
    print("\nGenerating plots...")
    fig1, fig2, fig3 = plot_voltage_sweep_results(results, V_array)
    
    print("\n[INFO] All plots generated.")
    print("[INFO] Close all figures to end the program.\n")
    
    plt.show()


if __name__ == "__main__":
    main()