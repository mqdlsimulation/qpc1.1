# simulations/qpc/simulation_split_trench_kwant.py

"""
simulation_split_trench_kwant.py

Kwant를 사용한 Split + Trench gate의 양자 수송 시뮬레이션.

Process:
1. Electrostatic potential φ(x,y) 계산 (기존 모듈)
2. Kwant tight-binding model 구성
3. Transmission T(E), Conductance G(E) 계산
4. Wavefunction |ψ|² 시각화
5. Conductance quantization 분석
"""

import numpy as np
import matplotlib.pyplot as plt

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
from modules.helper.plot.plot_kwant_transport import (
    plot_transmission,
    plot_conductance,
    plot_transmission_and_conductance,
    plot_wavefunction_2d,
    plot_multiple_wavefunctions,
    print_transport_analysis,
)


def main() -> None:
    print("=" * 70)
    print("Split + Trench Gate: Kwant Transport Simulation")
    print("Quantum Conductance Calculation")
    print("=" * 70)
    
    # ========== 2DEG 파라미터 출력 ==========
    print_2deg_parameters()
    
    # ========== 시뮬레이션 파라미터 ==========
    
    # Gate geometry
    gap = 80.0  # nm
    depth_d = 60.0  # nm, 2DEG depth
    
    # Grid 설정 (electrostatic potential용)
    x_range = (-300.0, 300.0)  # nm
    y_range = (-300.0, 300.0)  # nm
    nx_elec, ny_elec = 501, 501
    
    # Voltage 설정
    SPLIT_VG_MV = -500.0   # mV
    TRENCH_VG_MV = 1700.0  # mV
    
    # Kwant 설정
    points_per_lambda = 8  # Fermi 파장당 격자점 개수
    
    # Scattering region size (Kwant용)
    L_x_kwant = 40.0  # nm, x 방향 길이
    L_y_kwant = 300.0  # nm, y 방향 폭
    W_lead = 80.0     # nm, lead 폭
    
    # Energy range for transport
    E_min = 0.0        # eV
    E_max = EF_2DEG_EV * 1.5  # eV
    n_energies = 100
    
    # Temperature
    temperature_K = 0.1  # K
    
    # Wavefunction 계산할 에너지들 (meV 단위로 지정)
    wf_energies_meV = [5.0, 8.0, 11.0, 14.0]  # meV
    
    print("\n" + "=" * 70)
    print("Simulation Parameters")
    print("=" * 70)
    print(f"Split gate gap:           {gap} nm")
    print(f"2DEG depth:               {depth_d} nm")
    print(f"Split gate voltage:       {SPLIT_VG_MV} mV")
    print(f"Trench gate voltage:      {TRENCH_VG_MV} mV")
    print(f"Scattering region (Kwant): {L_x_kwant} × {L_y_kwant} nm")
    print(f"Lead width:               {W_lead} nm")
    print(f"Energy range:             [{E_min*1e3:.1f}, {E_max*1e3:.1f}] meV")
    print(f"Temperature:              {temperature_K} K")
    print("=" * 70)
    
    # ========== 1) Gate 구조 생성 ==========
    print("\n[Step 1] Building gate structures...")
    
    # Split gate
    split_gate = build_split_gate_structure(
        gap=gap,
        gate_width_x=500.0,
        gate_length_y=80.0,
        two_deg_depth=depth_d,
        use_dut_offset=False,
        do_describe=False,
        do_plot=False,
    )
    
    # Trench gate
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
    
    # Combined structure
    structure = SplitTrenchStructure(
        split_gate=split_gate,
        trench_gate=trench_gate,
    )
    
    print(f"  Split gate gap: {gap} nm")
    print(f"  Trench gate size: 78 × 500 nm")
    
    # ========== 2) Voltage 설정 ==========
    print("\n[Step 2] Configuring voltages...")
    
    split_volts = create_splitgate_voltages_from_mV(
        mode="symmetric",
        symmetric_Vg_mV=SPLIT_VG_MV,
        V_left_mV=0.0,
        V_right_mV=0.0,
    )
    
    trench_volts = create_trenchgate_voltages_from_mV(TRENCH_VG_MV)
    
    voltages = SplitTrenchVoltages(
        split_voltages=split_volts,
        trench_voltages=trench_volts,
    )
    
    print(f"  V_split (both):  {SPLIT_VG_MV} mV")
    print(f"  V_trench:        {TRENCH_VG_MV} mV")
    
    # ========== 3) Electrostatic potential 계산 ==========
    print("\n[Step 3] Computing electrostatic potential...")
    
    grid = make_uniform_grid(
        x_min=x_range[0],
        x_max=x_range[1],
        y_min=y_range[0],
        y_max=y_range[1],
        nx=nx_elec,
        ny=ny_elec,
        depth_d=depth_d,
    )
    
    pot_map = compute_split_trench_potential(
        structure=structure,
        voltages=voltages,
        grid=grid,
        screened=False,
    )
    
    phi_2d = pot_map.phi  # shape: (ny, nx), in eV
    
    print(f"  Potential range: [{np.min(phi_2d)*1e3:.1f}, {np.max(phi_2d)*1e3:.1f}] meV")
    
    # Grid 좌표 추출
    grid_x = grid.x  # 1D array
    grid_y = grid.y  # 1D array
    
    # ========== 4) Kwant 시스템 설정 ==========
    print("\n[Step 4] Setting up Kwant system...")
    
    # 격자 상수 추정
    a = estimate_lattice_constant(
        lambda_f_nm=LAMBDA_F_2DEG_NM,
        ef_ev=EF_2DEG_EV,
        target_points_per_wavelength=points_per_lambda,
    )
    
    # Hopping energy
    t = hopping_from_ef_lambda(a, EF_2DEG_EV, LAMBDA_F_2DEG_NM)
    print(f"  Hopping energy t = {t*1e3:.3f} meV")
    
    # Kwant config
    config = KwantSystemConfig(
        a=a,
        L_x=L_x_kwant,
        L_y=L_y_kwant,
        W_lead=W_lead,
        t=t,
        m_eff_kg=M_EFF_GAAS_KG,
        E_min=E_min,
        E_max=E_max,
        n_energies=n_energies,
    )
    
    print(f"  System: {L_x_kwant} × {L_y_kwant} nm")
    print(f"  Lead width: {W_lead} nm")
    print(f"  Lattice sites: ~{int(L_x_kwant/a)} × {int(L_y_kwant/a)}")
    
    # ========== 5) Transport 계산 ==========
    print("\n[Step 5] Running Kwant transport calculation...")
    
    # Wavefunction 계산할 에너지 (eV 변환)
    wf_energies_eV = [E * 1e-3 for E in wf_energies_meV]
    
    result = run_transport_calculation(
        config=config,
        potential_2d=phi_2d,
        grid_x=grid_x,
        grid_y=grid_y,
        fermi_energy=EF_2DEG_EV,
        compute_wf_at_energies=wf_energies_eV,
        verbose=True,
    )
    
    # ========== 6) 분석 결과 출력 ==========
    print_transport_analysis(result, temperature_K=temperature_K)
    
    # ========== 7) 시각화 ==========
    print("\n[Step 6] Generating plots...")
    
    # Plot 1: Transmission and Conductance (side by side)
    plot_transmission_and_conductance(
        result,
        highlight_ef=True,
        title=f"Transport Properties (gap={gap} nm, V_split={SPLIT_VG_MV} mV, V_trench={TRENCH_VG_MV} mV)",
    )
    
    # Plot 2: Transmission (detailed)
    plot_transmission(
        result,
        highlight_ef=True,
        title="Transmission vs Energy",
    )
    
    # Plot 3: Conductance (detailed)
    plot_conductance(
        result,
        highlight_ef=True,
        use_SI_units=False,
        title="Conductance vs Energy",
    )
    
    # Plot 4: Conductance (SI units)
    plot_conductance(
        result,
        highlight_ef=True,
        use_SI_units=True,
        title="Conductance vs Energy (SI units)",
    )
    
    # Plot 5: 2D potential map
    fig, ax = plt.subplots(figsize=(9, 7))
    
    X, Y = grid.meshgrid()
    
    im = ax.imshow(
        phi_2d * 1e3,  # meV
        origin='lower',
        extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
        cmap='RdBu_r',
        aspect='auto',
    )
    
    # Kwant scattering region 표시
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (0, -L_y_kwant/2),
        L_x_kwant,
        L_y_kwant,
        linewidth=2,
        edgecolor='lime',
        facecolor='none',
        label='Kwant scattering region'
    )
    ax.add_patch(rect)
    
    # Lead region 표시
    rect_lead = Rectangle(
        (0, -W_lead/2),
        L_x_kwant,
        W_lead,
        linewidth=2,
        edgecolor='yellow',
        facecolor='none',
        linestyle='--',
        label='Lead width'
    )
    ax.add_patch(rect_lead)
    
    ax.set_xlabel('x [nm]', fontsize=12)
    ax.set_ylabel('y [nm]', fontsize=12)
    ax.set_title('2D Electrostatic Potential + Kwant System', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Potential [meV]', fontsize=11)
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    # Plot 6: Wavefunctions (if calculated)
    if result.wf_data is not None and len(result.wf_data) > 0:
        print(f"  Plotting {len(result.wf_data)} wavefunctions...")
        
        plot_multiple_wavefunctions(
            result,
            max_plots=4,
            cmap='hot',
        )
        
        # 첫 번째 파동함수 상세 플롯
        if len(result.wf_data) > 0:
            pos, wf_dens = result.wf_data[0]
            E_meV = result.wf_energies[0] * 1e3
            
            plot_wavefunction_2d(
                positions=pos,
                wf_density=wf_dens,
                energy_meV=E_meV,
                title=f'Wavefunction at E = {E_meV:.2f} meV (Detailed)',
                use_log_scale=False,
            )
    
    print("\n[INFO] All plots generated.")
    print("[INFO] Close all figures to end the program.\n")
    
    plt.show()


if __name__ == "__main__":
    main()