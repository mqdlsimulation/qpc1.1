# simulations/qpc/simulation_subband_fixed_yaxis.py

"""
simulation_subband_fixed_yaxis.py

여러 (split_gap, trench_width) 조합에 대한 X 방향 subband 계산.

특징:
- Fermi energy = 8.1 meV 고정
- 모든 subplot의 Y축 범위 고정 (비교 용이)
- 각 subplot에 gap 위치 표시
- Split gap과 Trench width를 쌍으로 설정
- 기존 electrostatics 코드 활용
"""

from typing import List, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from modules.gate_structures.splitgate import (
    build_split_gate_structure,
    GateStructure,
)
from modules.gate_structures.trenchgate import (
    build_trench_gate_structure,
    TrenchGateStructure,
)
from modules.gate_voltages.voltages_splitgate import (
    create_splitgate_voltages_from_mV,
)
from modules.gate_voltages.voltages_trenchgate import (
    create_trenchgate_voltages_from_mV,
)
from modules.electrostatics.electrostatics_core import (
    ElectrostaticsGrid,
    make_uniform_grid,
)
from modules.electrostatics.electrostatics_split_trench import (
    SplitTrenchStructure,
    SplitTrenchVoltages,
    compute_split_trench_potential,
)
from modules.constants.physics import (
    M_EFF_GAAS_KG,
    print_2deg_parameters,
)

# 1D Schrödinger solver
from modules.quantum.schrodinger_1d import (
    solve_schrodinger_1d,
    print_subband_analysis,
)

from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------
# 1. 전압 설정
# ---------------------------------------------------------

def configure_voltages() -> SplitTrenchVoltages:
    """Split gate + Trench gate 전압 설정."""
    
    SPLIT_VG_MV: float = -100.0
    TRENCH_VG_MV: float = 2000
    
    split_volts = create_splitgate_voltages_from_mV(
        mode="symmetric",
        symmetric_Vg_mV=SPLIT_VG_MV,
        V_left_mV=0.0,
        V_right_mV=0.0,
    )
    
    trench_volts = create_trenchgate_voltages_from_mV(TRENCH_VG_MV)
    
    return SplitTrenchVoltages(
        split_voltages=split_volts,
        trench_voltages=trench_volts,
    )

SHOW_MAX_POINT_MULTI = True    # 다중 DEPTH_D 플롯에서 각 곡선의 최고점 표시

# ---------------------------------------------------------
# 2. X 방향 1D cut 추출
# ---------------------------------------------------------

def extract_1d_potential_x(
    phi_2d: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    y_position: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D potential에서 y=y_position에서의 V(x) 추출.
    
    반환:
        (x, V_1d)
    """
    iy = np.argmin(np.abs(grid_y - y_position))
    V_1d = phi_2d[iy, :]
    return grid_x, V_1d


# ---------------------------------------------------------
# 3. 여러 (gap, trench_width) 조합 처리
# ---------------------------------------------------------

def compute_all_configurations(
    config_pairs: List[Tuple[float, float]],
    depth_d: float,
    grid: ElectrostaticsGrid,
    voltages: SplitTrenchVoltages,
    y_position: float = 0.0,
    n_states: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[dict]]:
    """
    여러 (split_gap, trench_width) 조합에 대해 계산.
    
    파라미터:
        config_pairs: [(gap1, trench1), (gap2, trench2), ...]
        ...
    
    반환:
        (x_arrays, V_arrays, energies_list, analysis_list)
    """
    
    x_arrays = []
    V_arrays = []
    energies_list = []
    analysis_list = []
    
    for idx, (gap, trench_width) in enumerate(config_pairs):
        print(f"\n[Config {idx+1}/{len(config_pairs)}] Gap={gap} nm, Trench={trench_width} nm")
        print("-" * 60)
        
        # Split gate 생성
        split_gate = build_split_gate_structure(
            gap=gap,
            gate_width_x=500.0,
            gate_length_y=80.0,
            two_deg_depth=depth_d,
            use_dut_offset=False,
            do_describe=False,
            do_plot=False,
        )
        
        # Trench gate 생성 (trench_width 사용)
        trench_gate = build_trench_gate_structure(
            x_length=trench_width,
            y_width=500.0,
            x_offset=0.0,
            y_offset=0.0,
            two_deg_depth=depth_d,
            split_shapes=split_gate.shapes,
            do_describe=False,
            do_plot=False,
            do_overlap_check=True,
        )
        
        # 결합 구조
        structure = SplitTrenchStructure(
            split_gate=split_gate,
            trench_gate=trench_gate,
        )
        
        # 전위 계산
        pot_map = compute_split_trench_potential(
            structure=structure,
            voltages=voltages,
            grid=grid,
            screened=False,
        )
        
        phi_2d = pot_map.phi
        
        print(f"  2D potential: [{np.min(phi_2d)*1e3:.1f}, {np.max(phi_2d)*1e3:.1f}] meV")
        
        # X 방향 1D potential 추출
        x, V_1d = extract_1d_potential_x(
            phi_2d=phi_2d,
            grid_x=grid.x,
            grid_y=grid.y,
            y_position=y_position,
        )
        
        print(f"  V(x) range: [{np.min(V_1d)*1e3:.1f}, {np.max(V_1d)*1e3:.1f}] meV")
        
        # 1D Schrödinger 풀기
        energies, _ = solve_schrodinger_1d(
            x=x,
            V=V_1d,
            m_eff_kg=M_EFF_GAAS_KG,
            n_states=n_states,
        )
        
        # 분석 (Fermi energy는 나중에 설정)
        analysis = {
            'gap': gap,
            'trench_width': trench_width,
            'energies_meV': energies * 1e3,
        }
        
        print(f"  E_0 = {energies[0]*1e3:.3f} meV")
        if len(energies) > 1:
            print(f"  E_1 = {energies[1]*1e3:.3f} meV")
            print(f"  ΔE = {(energies[1]-energies[0])*1e3:.3f} meV")
        
        x_arrays.append(x)
        V_arrays.append(V_1d)
        energies_list.append(energies)
        analysis_list.append(analysis)
    
    return x_arrays, V_arrays, energies_list, analysis_list


# ---------------------------------------------------------
# 4. Y축 고정 subplot 시각화
# ---------------------------------------------------------

def plot_fixed_yaxis_comparison(
    config_pairs: List[Tuple[float, float]],
    x_arrays: List[np.ndarray],
    V_arrays: List[np.ndarray],
    energies_list: List[np.ndarray],
    # E_fermi: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    n_levels: int = 5,
    figsize: Tuple[float, float] = (16, 10),
) -> None:
    """
    모든 subplot의 Y축을 고정하고, 각 gap 위치를 표시.
    
    파라미터:
        config_pairs: [(gap, trench_width), ...]
        x_arrays, V_arrays, energies_list: 계산 결과
        E_fermi: Fermi energy [eV]
        xlim: X축 범위
        ylim: Y축 범위 (고정)
        n_levels: 표시할 subband 개수
        figsize: Figure 크기
    """
    
    n_configs = len(config_pairs)
    
    # Subplot layout
    ncols = min(3, n_configs)
    nrows = (n_configs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if n_configs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Level colors
    level_colors = plt.cm.rainbow(np.linspace(0, 1, n_levels))
    
    # eV → meV
    # E_f_meV = E_fermi * 1e3
    
    for idx, (config_pair, x, V, energies, ax) in enumerate(
        zip(config_pairs, x_arrays, V_arrays, energies_list, axes)
    ):
        gap, trench_width = config_pair
        
        # X 범위 마스크
        if xlim is not None:
            mask = (x >= xlim[0]) & (x <= xlim[1])
            x_plot = x[mask]
            V_plot = V[mask]
        else:
            x_plot = x
            V_plot = V
        
        V_meV = V_plot * 1e3
        energies_meV = energies * 1e3
        
        # --- subband spacing ΔE = E1 - E0 계산 (추가 부분) ---
        if len(energies_meV) > 1:
            deltaE_meV = energies_meV[1] - energies_meV[0]
        else:
            deltaE_meV = np.nan
        # --------------------------------------------------
        
        # Potential curve
        ax.plot(x_plot, V_meV, 'b-', linewidth=2.5, label='V(x)', zorder=1)
        
        # Energy levels
        n_to_show = min(n_levels, len(energies))
        
        for level_idx in range(n_to_show):
            E_meV = energies_meV[level_idx]
            color = level_colors[level_idx]
            
            # 수평선
            # ax.axhline(E_meV, color=color, linestyle='--',
            #           linewidth=2, alpha=0.7, zorder=2)
            
            # Label (왼쪽에)
            # ax.text(x_plot[0] + 5, E_meV,
            #        f'$E_{level_idx}$',
            #        va='center', ha='left', fontsize=9,
            #        color=color, fontweight='bold',
            #        bbox=dict(boxstyle='round,pad=0.3',
            #                 facecolor='white',
            #                 edgecolor=color, alpha=0.8))
        
        # Fermi energy
        # ax.axhline(E_f_meV, color='red', linestyle='-',
        #           linewidth=3, alpha=0.8, zorder=5)
        
        # Count modes - 모드 개수
        # n_modes = np.sum(energies < E_fermi)
        # ax.text(x_plot[-1] - 5, E_f_meV,
        #        f'{n_modes} modes  ',
        #        va='bottom', ha='right', fontsize=10,
        #        color='red', fontweight='bold')
        
        # --- 각 subplot에 ΔE 텍스트 표시 (추가 부분) ---
        if not np.isnan(deltaE_meV):
            ax.text(
                x_plot[-1] - 5,
                ylim[0] + 100,   # y축 아래쪽에 배치, y축 좌표값 기준
                f'ΔE = {deltaE_meV:.2f} meV',
                va='bottom',
                ha='right',
                fontsize=16,
                color='black',
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='black',
                    alpha=0.8,
                ),
            )
        # ------------------------------------------------
        
        # Gap boundaries (회색 수직선)
        # ax.axvline(-gap/2, color='gray', linestyle=':',
        #           alpha=0.6, linewidth=2.5, zorder=4)
        # ax.axvline(+gap/2, color='gray', linestyle=':',
        #           alpha=0.6, linewidth=2.5, zorder=4)
        
        # Gap labels - 갭 라벨
        # ax.text(-gap/2, ylim[1] * 0.98, f'  -{gap/2:.0f}',
        #        va='top', ha='left', fontsize=9,
        #        color='gray', fontweight='bold')
        # ax.text(+gap/2, ylim[1] * 0.98, f'  +{gap/2:.0f}',
        #        va='top', ha='left', fontsize=9,
        #        color='gray', fontweight='bold')
        
        # Formatting
        ax.set_xlabel('x [nm]', fontsize=18, fontweight='bold')
        ax.set_ylabel('Energy [meV]', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(f'Gap={gap:.0f} nm, Trench={trench_width:.0f} nm',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # X, Y 범위 고정
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    
    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()

# ---------------------------------------------------------
# 4-1. 모든 gap의 V(x) + ΔE(E0,E1) 통합 플롯
# ---------------------------------------------------------

def plot_unified_potentials_with_deltaE(
    config_pairs: List[Tuple[float, float]],
    x_arrays: List[np.ndarray],
    V_arrays: List[np.ndarray],
    energies_list: List[np.ndarray],
    # E_fermi: float,
    tick_fontsize: int = 20,          # ✅ 추가: 축 숫자 크기
    tick_fontweight: str = "normal",    # ✅ 추가: 축 숫자 굵기(원치 않으면 "normal")
    xlim: Tuple[float, float] | None = None,
    title: str = "V(x, y=0) for all gaps with ΔE = E1 - E0",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    simulation_schrodinger_1d_subband.py 스타일로,
    하나의 figure 에 모든 gap 의 V(x) 곡선을 그리고
    각 gap 에 대해 E0, E1 및 ΔE(E1 - E0)를 표시.
    """

    fig, ax = plt.subplots(figsize=(14, 9))

    n_configs = len(config_pairs)
    colors = plt.cm.tab10(np.linspace(0, 1, n_configs))

    for (gap, trench_width), x, V, energies, color in zip(
        config_pairs, x_arrays, V_arrays, energies_list, colors
    ):
        # X 범위 마스크
        if xlim is not None:
            mask = (x >= xlim[0]) & (x <= xlim[1])
            x_plot = x[mask]
            V_plot = V[mask]
        else:
            x_plot = x
            V_plot = V

        V_meV = V_plot * 1e3
        energies_meV = energies * 1e3

        # ΔE 계산 (E1 - E0)
        if len(energies_meV) > 1:
            E0_meV = energies_meV[0]
            E1_meV = energies_meV[1]
            ΔE_meV = E1_meV - E0_meV
            label = f"Gap {gap:.0f} nm (ΔE={ΔE_meV:.2f} meV)"
        else:
            E0_meV = energies_meV[0]
            E1_meV = np.nan
            ΔE_meV = np.nan
            label = f"Gap {gap:.0f} nm"

        # V(x) 곡선
        ax.plot(
            x_plot,
            V_meV,
            "-",
            linewidth=2.5,
            color=color,
            alpha=0.85,
            label=label,
        )

        # E0, E1 level 수평선 + ΔE 텍스트 (gap 주변에서만 표시)
        # if len(energies_meV) > 1:
        #     if xlim is not None:
        #         x_level_min = max(xlim[0], -gap / 2 - 10.0)
        #         x_level_max = min(xlim[1], +gap / 2 + 10.0)
        #     else:
        #         x_level_min = -gap / 2 - 10.0
        #         x_level_max = +gap / 2 + 10.0

        #     ax.hlines(
        #         E0_meV,
        #         x_level_min,
        #         x_level_max,
        #         colors=color,
        #         linestyles="--",
        #         linewidth=1.5,
        #         alpha=0.7,
        #     )
        #     ax.hlines(
        #         E1_meV,
        #         x_level_min,
        #         x_level_max,
        #         colors=color,
        #         linestyles="--",
        #         linewidth=1.5,
        #         alpha=0.7,
        #     )

        #     ax.text(
        #         x_level_max + 2.0,
        #         0.5 * (E0_meV + E1_meV),
        #         f"ΔE={ΔE_meV:.2f} meV",
        #         va="center",
        #         ha="left",
        #         fontsize=9,
        #         color=color,
        #         fontweight="bold",
        #     )

    # Fermi energy 라인
    # if E_fermi is not None:
    #     E_f_meV = E_fermi * 1e3
    #     ax.axhline(
    #         E_f_meV,
    #         color="red",
    #         linestyle="-",
    #         linewidth=3,
    #         alpha=0.8,
    #         label=f"$E_F$ = {E_f_meV:.1f} meV",
    #     )

    # 축/레이블/범례
    ax.set_xlabel("x [nm]", fontsize=18, fontweight="bold")
    ax.set_ylabel("Energy [meV]", fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=18, framealpha=0.9)
    # ✅ 축 숫자(틱 라벨) 크기/굵기 키우기
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
    plt.setp(ax.get_xticklabels(), fontweight=tick_fontweight)
    plt.setp(ax.get_yticklabels(), fontweight=tick_fontweight)

    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    return fig, ax

# ---------------------------------------------------------
# 5. Summary table 생성
# ---------------------------------------------------------

def print_summary_table(
    config_pairs: List[Tuple[float, float]],
    energies_list: List[np.ndarray],
    # E_fermi: float,
) -> None:
    """
    각 configuration의 subband 정보를 table로 출력.
    """
    
    print(f"\n{'='*80}")
    print("Summary: Subband Analysis for Each Configuration")
    print(f"{'='*80}")
    print(f"{'Gap [nm]':<12} {'Trench [nm]':<14} {'E_0 [meV]':<12} "
          f"{'ΔE [meV]':<12} {'Modes':<8}")
    print("-" * 80)
    
    # E_f_meV = E_fermi * 1e3
    
    for config_pair, energies in zip(config_pairs, energies_list):
        gap, trench_width = config_pair
        energies_meV = energies * 1e3
        
        E_0 = energies_meV[0]
        
        if len(energies) > 1:
            ΔE = energies_meV[1] - energies_meV[0]
        else:
            ΔE = np.nan
        
        # n_modes = np.sum(energies < E_fermi)
        
        print(f"{gap:<12.0f} {trench_width:<14.0f} {E_0:<12.3f} ")
            #   f"{ΔE:<12.3f} {n_modes:<8d}")
    
    print("=" * 80)
    # print(f"Fermi energy: {E_f_meV:.3f} meV")
    print("=" * 80)

def plot_deltaE_vs_gap_multi_depth(
    config_pairs: List[Tuple[float, float]],
    depth_d_list: List[float],
    grid: ElectrostaticsGrid,
    voltages: SplitTrenchVoltages,
    y_position: float = 0.0,
    n_states: int = 10,
    use_smooth_curve: bool = True,
    show_markers: bool = False,
    figsize: Tuple[float, float] = (12, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    여러 DEPTH_D 값에 대해 ΔE vs Gap을 동일한 plot에 표시.
    
    파라미터:
        config_pairs: [(gap, trench_width), ...]
        depth_d_list: [depth1, depth2, ...] DEPTH_D 값 리스트
        grid: ElectrostaticsGrid (depth_d는 각 iteration에서 업데이트)
        voltages: SplitTrenchVoltages
        ...
    
    반환:
        (fig, ax)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 색상 자동 지정
    colors = plt.cm.tab10(np.linspace(0, 1, len(depth_d_list)))
    
    gaps = [pair[0] for pair in config_pairs]
    
    for depth_d, color in zip(depth_d_list, colors):
        print(f"\n{'='*60}")
        print(f"Computing for DEPTH_D = {depth_d} nm...")
        print(f"{'='*60}")
        
        # 해당 depth_d에 대해 grid 재생성
        grid_temp = make_uniform_grid(
            x_min=grid.x.min(),
            x_max=grid.x.max(),
            y_min=grid.y.min(),
            y_max=grid.y.max(),
            nx=len(grid.x),
            ny=len(grid.y),
            depth_d=depth_d,
        )
        
        # 모든 configuration 계산
        _, _, energies_list, _ = compute_all_configurations(
            config_pairs=config_pairs,
            depth_d=depth_d,
            grid=grid_temp,
            voltages=voltages,
            y_position=y_position,
            n_states=n_states,
        )
        
        # ΔE 계산
        ΔE_values = []
        for energies in energies_list:
            if len(energies) > 1:
                ΔE = (energies[1] - energies[0]) * 1e3  # meV
            else:
                ΔE = np.nan
            ΔE_values.append(ΔE)
        
        # 플롯
        if use_smooth_curve:
            gaps_arr = np.array(gaps)
            ΔE_arr = np.array(ΔE_values)
            
            valid_mask = ~np.isnan(ΔE_arr)
            gaps_valid = gaps_arr[valid_mask]
            ΔE_valid = ΔE_arr[valid_mask]
            
            # 중복 x값 제거
            unique_gaps, unique_indices = np.unique(gaps_valid, return_inverse=True)
            ΔE_unique = np.array([ΔE_valid[unique_indices == i].mean() for i in range(len(unique_gaps))])
            
            if len(unique_gaps) >= 2:
                gaps_smooth = np.linspace(unique_gaps.min(), unique_gaps.max(), 300)
                spl = make_interp_spline(unique_gaps, ΔE_unique, k=min(3, len(unique_gaps)-1))
                ΔE_smooth = spl(gaps_smooth)
                
                ax.plot(
                    gaps_smooth,
                    ΔE_smooth,
                    '-',
                    color=color,
                    linewidth=2.5,
                    label=f'd = {depth_d:.0f} nm',
                )
            
            if show_markers:
                ax.plot(gaps_valid, ΔE_valid, 'o', color=color, markersize=8)
            
            # ★ 최고점 표기 (use_smooth_curve=True일 때)
            if SHOW_MAX_POINT_MULTI and len(ΔE_unique) > 0:
                max_idx = np.argmax(ΔE_unique)
                max_gap = unique_gaps[max_idx]
                max_ΔE = ΔE_unique[max_idx]
                
                ax.plot(max_gap, max_ΔE, 'o', color=color, markersize=10,
                        markeredgewidth=1.5, zorder=5)
                
                ax.text(
                    max_gap,
                    max_ΔE + 0.5,
                    f'({max_gap:.0f}, {max_ΔE:.2f})',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    color=color,
                )
        
        else:
            ax.plot(
                gaps,
                ΔE_values,
                'o-',
                color=color,
                linewidth=2.5,
                markersize=8,
                label=f'd = {depth_d:.0f} nm',
            )
            
            # ★ 최고점 표기 (use_smooth_curve=False일 때)
            if SHOW_MAX_POINT_MULTI:
                ΔE_arr = np.array(ΔE_values)
                valid_mask = ~np.isnan(ΔE_arr)
                gaps_valid = np.array(gaps)[valid_mask]
                ΔE_valid = ΔE_arr[valid_mask]
                
                if len(ΔE_valid) > 0:
                    max_idx = np.argmax(ΔE_valid)
                    max_gap = gaps_valid[max_idx]
                    max_ΔE = ΔE_valid[max_idx]
                    
                    ax.plot(max_gap, max_ΔE, 'o', color=color, markersize=12,
                            markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                    
                    ax.text(
                        max_gap,
                        max_ΔE + 0.5,
                        f'({max_gap:.0f}, {max_ΔE:.2f})',
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold',
                        color=color,
                    )
    
    # 축 설정
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('Split Gate Gap [nm]', fontsize=15, fontweight='bold')
    ax.set_ylabel('ΔE = E$_1$ - E$_0$ [meV]', fontsize=15, fontweight='bold')
    ax.set_title('ΔE vs Gap for Various 2DEG Depths',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax

# ---------------------------------------------------------
# 6. Main
# ---------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Multi-Configuration Subband Calculation")
    print("Fixed Y-axis, Gap markers, Split-Trench pairs")
    print("=" * 70)
    
    print_2deg_parameters()
    
    # ========== Parameters ==========
    # Configuration pairs: (split_gap, trench_width)
    CONFIG_PAIRS = [
        (10, 8),
        (20, 18),
        (30, 28),
        (40, 38),
        (50, 48),
        (60, 58),
        (70, 68),
        (80, 78),
        (90, 88),
        (110, 108),
        (120, 118),
        (130, 128),
    ]

    # ================================
    # ★ Plot 스위치 설정 (on/off)
    # ================================
    SHOW_FERMI_LINE = False        # ΔE vs Gap 그래프에서 Fermi 에너지 점선 표시 여부
    SHOW_ALL_LABELS = False        # 모든 포인트에 값 표시 (False면 최대값만 표시)
    USE_SMOOTH_CURVE = True        # 부드러운 곡선으로 연결 (False면 직선+마커)
    SHOW_MARKERS = False           # 데이터 포인트 마커 표시 여부
    MULTI_DEPTH_PLOT = True        # 여러 DEPTH_D 값 비교 플롯 생성


    # Fermi energy (사용자 지정)
    E_FERMI_EV = 8.1e-3  # 8.1 meV = 0.0081 eV
    
    DEPTH_D = 30.0  # nm
    DEPTH_D_LIST = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # 비교할 DEPTH_D 값들 [nm]

    # Grid
    X_RANGE = (-300.0, 300.0)
    Y_RANGE = (-300.0, 300.0)
    NX = 501
    NY = 501
    
    # 1D analysis
    Y_POSITION = 0.0  # y=0에서 x 방향 cut
    N_STATES = 10
    
    # Plot
    XLIM_PLOT = (-300, 300)
    YLIM_PLOT = (-700, 20)  # Y축 고정 범위
    N_LEVELS_PLOT = 5
    # ================================
    
    print(f"\nSimulation parameters:")
    print(f"  Configurations: {len(CONFIG_PAIRS)} pairs")
    for gap, trench in CONFIG_PAIRS:
        print(f"    - Gap={gap} nm, Trench={trench} nm")
    # print(f"  Fermi energy: {E_FERMI_EV*1e3:.3f} meV")
    print(f"  2DEG depth: {DEPTH_D} nm")
    print(f"  X-direction cut at: y = {Y_POSITION} nm")
    print(f"  Y-axis range: {YLIM_PLOT} meV")
    
    # ========== 1) 전압 설정 ==========
    print(f"\n{'='*70}")
    print("Step 1: Configuring voltages...")
    print(f"{'='*70}")
    
    voltages = configure_voltages()
    voltages.describe()
    
    # ========== 2) Grid 생성 ==========
    print(f"\n{'='*70}")
    print("Step 2: Creating grid...")
    print(f"{'='*70}")
    
    grid = make_uniform_grid(
        x_min=X_RANGE[0],
        x_max=X_RANGE[1],
        y_min=Y_RANGE[0],
        y_max=Y_RANGE[1],
        nx=NX,
        ny=NY,
        depth_d=DEPTH_D,
    )
    
    print(f"Grid: {len(grid.x)} × {len(grid.y)} = {len(grid.x) * len(grid.y)} points")
    
    # ========== 3) 모든 configuration 계산 ==========
    print(f"\n{'='*70}")
    print("Step 3: Computing all configurations...")
    print(f"{'='*70}")
    
    x_arrays, V_arrays, energies_list, analysis_list = compute_all_configurations(
        config_pairs=CONFIG_PAIRS,
        depth_d=DEPTH_D,
        grid=grid,
        voltages=voltages,
        y_position=Y_POSITION,
        n_states=N_STATES,
    )
    
    # ========== 4) Summary table ==========
    print_summary_table(
        config_pairs=CONFIG_PAIRS,
        energies_list=energies_list,
        # E_fermi=E_FERMI_EV,
    )
    
    # ========== 5) 시각화 ==========
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    output_dir = Path("output/subband_fixed_yaxis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unified plot: 모든 gap 의 V(x) + E0, E1, ΔE 표시
    print("  Creating unified potential plot with ΔE annotations...")
    fig_unified, ax_unified = plot_unified_potentials_with_deltaE(
        config_pairs=CONFIG_PAIRS,
        x_arrays=x_arrays,
        V_arrays=V_arrays,
        energies_list=energies_list,
        # E_fermi=E_FERMI_EV,
        xlim=XLIM_PLOT,
        title="V(x, y=0) for all gaps with E0, E1 and ΔE",
    )
    
    filename_unified = output_dir / "potentials_with_deltaE_all_gaps.png"
    fig_unified.savefig(filename_unified, dpi=200, bbox_inches="tight")
    print(f"  Saved: {filename_unified}")
    
    # Main plot: Fixed Y-axis comparison  (★ 한 번만 호출)
    print("  Creating fixed Y-axis comparison plot...")
    plot_fixed_yaxis_comparison(
        config_pairs=CONFIG_PAIRS,
        x_arrays=x_arrays,
        V_arrays=V_arrays,
        energies_list=energies_list,
        # E_fermi=E_FERMI_EV,
        xlim=XLIM_PLOT,
        ylim=YLIM_PLOT,
        n_levels=N_LEVELS_PLOT,
    )
    
    filename_main = output_dir / "comparison_fixed_yaxis.png"
    plt.savefig(filename_main, dpi=200, bbox_inches='tight')
    print(f"  Saved: {filename_main}")

    
    # ΔE vs Gap plot

    print("  Creating ΔE vs Gap trend plot...")
    fig_trend, ax_trend = plt.subplots(figsize=(10, 7))
    
    gaps = [pair[0] for pair in CONFIG_PAIRS]
    ΔE_values = []
    
    for energies in energies_list:
        if len(energies) > 1:
            ΔE = (energies[1] - energies[0]) * 1e3  # E1 - E0 [meV]
        else:
            ΔE = np.nan
        ΔE_values.append(ΔE)
    
# 기본 곡선
    if USE_SMOOTH_CURVE:
        # 부드러운 곡선 (스플라인 보간)
        gaps_arr = np.array(gaps)
        ΔE_arr = np.array(ΔE_values)
        
        # NaN 제거
        valid_mask = ~np.isnan(ΔE_arr)
        gaps_valid = gaps_arr[valid_mask]
        ΔE_valid = ΔE_arr[valid_mask]
        
        # 중복 x값 제거 (평균값 사용)
        unique_gaps, unique_indices = np.unique(gaps_valid, return_inverse=True)
        ΔE_unique = np.array([ΔE_valid[unique_indices == i].mean() for i in range(len(unique_gaps))])
        
        if len(unique_gaps) >= 2:
            # 스플라인 보간
            gaps_smooth = np.linspace(unique_gaps.min(), unique_gaps.max(), 300)
            spl = make_interp_spline(unique_gaps, ΔE_unique, k=min(3, len(unique_gaps)-1))
            ΔE_smooth = spl(gaps_smooth)
            
            ax_trend.plot(
                gaps_smooth,
                ΔE_smooth,
                'b-',
                linewidth=2.5,
                label='ΔE (calculated)',
            )
        
        # 마커 표시 (옵션)
        if SHOW_MARKERS:
            ax_trend.plot(
                gaps_valid,
                ΔE_valid,
                'bo',
                markersize=10,
            )
    else:
        # 기존 방식 (직선 + 마커)
        ax_trend.plot(
            gaps,
            ΔE_values,
            'bo-',
            linewidth=2.5,
            markersize=10,
            label='ΔE (calculated)',
        )


    # 각 Gap 포인트 위에 ΔE 값 숫자로 표시 (스위치 적용)
    if SHOW_ALL_LABELS:
        # 모든 포인트에 값 표시
        for gap, ΔE in zip(gaps, ΔE_values):
            if np.isnan(ΔE):
                continue
            ax_trend.text(
                gap,
                ΔE - 1, # 포인트 라벨 위치 조정
                f"{ΔE:.2f}",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
            )
    else:
        # 최대 y값만 표시
        valid_indices = [i for i, ΔE in enumerate(ΔE_values) if not np.isnan(ΔE)]
        if valid_indices:
            max_idx = max(valid_indices, key=lambda i: ΔE_values[i])
            ax_trend.text(
                gaps[max_idx],
                ΔE_values[max_idx] - 2,
                f"{ΔE_values[max_idx]:.2f}",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
            )
    
    # Fermi energy reference (스위치 적용)
    if SHOW_FERMI_LINE:
        ax_trend.axhline(
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
        )

    # ========== 다중 DEPTH_D 비교 플롯 ==========
    if MULTI_DEPTH_PLOT:
        print(f"\n{'='*70}")
        print("Creating Multi-DEPTH_D comparison plot...")
        print(f"{'='*70}")
        
        fig_multi, ax_multi = plot_deltaE_vs_gap_multi_depth(
            config_pairs=CONFIG_PAIRS,
            depth_d_list=DEPTH_D_LIST,
            grid=grid,
            voltages=voltages,
            y_position=Y_POSITION,
            n_states=N_STATES,
            use_smooth_curve=USE_SMOOTH_CURVE,
            show_markers=SHOW_MARKERS,
        )
        
        filename_multi = output_dir / "deltaE_vs_gap_multi_depth.png"
        fig_multi.savefig(filename_multi, dpi=200, bbox_inches='tight')
        print(f"  Saved: {filename_multi}")

    ax_trend.tick_params(axis='x', labelsize=18)
    ax_trend.tick_params(axis='y', labelsize=18)
    ax_trend.set_xlabel('Split Gate Gap [nm]', fontsize=15, fontweight='bold')
    ax_trend.set_ylabel('ΔE = E$_1$ - E$_0$ [meV]', fontsize=15, fontweight='bold')
    ax_trend.set_title('ΔE vs Gap\n(X-direction confinement)',
                       fontsize=15, fontweight='bold')
    ax_trend.legend(fontsize=17, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    ax_trend.grid(True, alpha=0.3)
    
    filename_trend = output_dir / "spacing_vs_gap.png"
    plt.savefig(filename_trend, dpi=200, bbox_inches='tight')
    print(f"  Saved: {filename_trend}")

    # ΔE vs Trench width plot
    print("  Creating ΔE vs Trench-width trend plot...")
    fig_trench, ax_trench = plt.subplots(figsize=(10, 7))
    
    trenches = [pair[1] for pair in CONFIG_PAIRS]  # 각 config 의 trench width [nm]
    
# 기본 곡선
    if USE_SMOOTH_CURVE:
        # 부드러운 곡선 (스플라인 보간)
        trenches_arr = np.array(trenches)
        ΔE_arr = np.array(ΔE_values)
        
        # NaN 제거
        valid_mask = ~np.isnan(ΔE_arr)
        trenches_valid = trenches_arr[valid_mask]
        ΔE_valid = ΔE_arr[valid_mask]
        
        # 중복 x값 제거 (평균값 사용)
        unique_trenches, unique_indices = np.unique(trenches_valid, return_inverse=True)
        ΔE_unique = np.array([ΔE_valid[unique_indices == i].mean() for i in range(len(unique_trenches))])
        
        if len(unique_trenches) >= 2:
            # 스플라인 보간
            trenches_smooth = np.linspace(unique_trenches.min(), unique_trenches.max(), 300)
            spl = make_interp_spline(unique_trenches, ΔE_unique, k=min(3, len(unique_trenches)-1))
            ΔE_smooth = spl(trenches_smooth)
            
            ax_trench.plot(
                trenches_smooth,
                ΔE_smooth,
                'm-',
                linewidth=2.5,
                label='ΔE (calculated)',
            )
        
        # 마커 표시 (옵션)
        if SHOW_MARKERS:
            ax_trench.plot(
                trenches_valid,
                ΔE_valid,
                'mo',
                markersize=10,
            )
    else:
        # 기존 방식 (직선 + 마커)
        ax_trench.plot(
            trenches,
            ΔE_values,
            'mo-',
            linewidth=2.5,
            markersize=10,
            label='ΔE (calculated)',
        )
    
    # 각 Trench 포인트 위에 ΔE 값 숫자로 표시 (스위치 적용)
    if SHOW_ALL_LABELS:
        for trench, ΔE in zip(trenches, ΔE_values):
            if np.isnan(ΔE):
                continue
            ax_trench.text(
                trench,
                ΔE + 0.1,
                f"{ΔE:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    else:
        valid_indices = [i for i, ΔE in enumerate(ΔE_values) if not np.isnan(ΔE)]
        if valid_indices:
            max_idx = max(valid_indices, key=lambda i: ΔE_values[i])
            ax_trench.text(
                trenches[max_idx],
                ΔE_values[max_idx] + 0.1,
                f"{ΔE_values[max_idx]:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    
    # (선택) Fermi energy 는 참고용으로 동일하게 추가
    # ax_trench.axhline(
    #     E_f_meV,
    #     color='red',
    #     linestyle='--',
    #     linewidth=2,
    #     alpha=0.7,
    #     label=f'$E_F$ = {E_f_meV:.1f} meV',
    # )
    
    ax_trench.set_xlabel('Trench Gate Width [nm]', fontsize=13, fontweight='bold')
    ax_trench.set_ylabel('ΔE = E$_1$ - E$_0$ [meV]', fontsize=13, fontweight='bold')
    ax_trench.set_title('ΔE vs Trench Width',
                        fontsize=15, fontweight='bold')
    ax_trench.legend(fontsize=11, loc='upper right')
    ax_trench.grid(True, alpha=0.3)
    
    filename_trench = output_dir / "spacing_vs_trench.png"
    plt.savefig(filename_trench, dpi=200, bbox_inches='tight')
    print(f"  Saved: {filename_trench}")


    
    # E_0 vs Gap plot
    print("  Creating E_0 vs Gap plot...")
    fig_e0, ax_e0 = plt.subplots(figsize=(10, 7))
    
    E_0_values = [energies[0] * 1e3 for energies in energies_list]
    
    if USE_SMOOTH_CURVE:
        gaps_arr = np.array(gaps)
        E0_arr = np.array(E_0_values)
        
        # 중복 x값 제거 (평균값 사용)
        unique_gaps, unique_indices = np.unique(gaps_arr, return_inverse=True)
        E0_unique = np.array([E0_arr[unique_indices == i].mean() for i in range(len(unique_gaps))])
        
        if len(unique_gaps) >= 2:
            gaps_smooth = np.linspace(unique_gaps.min(), unique_gaps.max(), 300)
            spl = make_interp_spline(unique_gaps, E0_unique, k=min(3, len(unique_gaps)-1))
            E0_smooth = spl(gaps_smooth)
            
            ax_e0.plot(
                gaps_smooth,
                E0_smooth,
                'g-',
                linewidth=2.5,
                label='$E_0$ (ground state)',
            )
        
        if SHOW_MARKERS:
            ax_e0.plot(gaps_arr, E0_arr, 'go', markersize=10)
    else:
        ax_e0.plot(gaps, E_0_values, 'go-',
                  linewidth=2.5, markersize=10, label='$E_0$ (ground state)')
    
    # ax_e0.axhline(E_f_meV, color='red', linestyle='--',
    #              linewidth=2, alpha=0.7,
    #              label=f'$E_F$ = {E_f_meV:.1f} meV')
    
    ax_e0.set_xlabel('Split Gate Gap [nm]', fontsize=15, fontweight='bold')
    ax_e0.set_ylabel('Ground State Energy $E_0$ [meV]', fontsize=15, fontweight='bold')
    ax_e0.set_title('Ground State Energy vs Gap',
                   fontsize=15, fontweight='bold')
    ax_e0.legend(fontsize=11, loc='upper left')
    ax_e0.grid(True, alpha=0.3)
    
    filename_e0 = output_dir / "E0_vs_gap.png"
    plt.savefig(filename_e0, dpi=200, bbox_inches='tight')
    print(f"  Saved: {filename_e0}")
    
    print(f"\n{'='*70}")
    print("Simulation completed successfully!")
    print(f"{'='*70}\n")
    
    plt.show()
    
    return {
        'config_pairs': CONFIG_PAIRS,
        'x_arrays': x_arrays,
        'V_arrays': V_arrays,
        'energies_list': energies_list,
        'analysis_list': analysis_list,
        # 'E_fermi': E_FERMI_EV,
    }


if __name__ == "__main__":
    results = main()