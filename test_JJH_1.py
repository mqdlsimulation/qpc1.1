# simulations/qpc/subband_miniplot.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from modules.gate_structures.splitgate import build_split_gate_structure
from modules.gate_structures.trenchgate import build_trench_gate_structure
from modules.gate_voltages.voltages_splitgate import create_splitgate_voltages_from_mV
from modules.gate_voltages.voltages_trenchgate import create_trenchgate_voltages_from_mV
from modules.electrostatics.electrostatics_core import ElectrostaticsGrid, make_uniform_grid
from modules.electrostatics.electrostatics_split_trench import (
    SplitTrenchStructure, SplitTrenchVoltages, compute_split_trench_potential
)
from modules.constants.physics import M_EFF_GAAS_KG, print_2deg_parameters
from modules.quantum.schrodinger_1d import solve_schrodinger_1d


# -----------------------------
# Data container
# -----------------------------
@dataclass
class Result:
    gap_nm: float
    trench_nm: float
    x_nm: np.ndarray
    V_eV: np.ndarray
    energies_eV: np.ndarray

    @property
    def E0_meV(self) -> float:
        return float(self.energies_eV[0] * 1e3)

    @property
    def E1_meV(self) -> float:
        return float(self.energies_eV[1] * 1e3) if len(self.energies_eV) > 1 else np.nan

    @property
    def dE_meV(self) -> float:
        return self.E1_meV - self.E0_meV if np.isfinite(self.E1_meV) else np.nan


# -----------------------------
# Config / helpers
# -----------------------------
def configure_voltages(split_Vg_mV: float = -200.0, trench_Vg_mV: float = 1000.0) -> SplitTrenchVoltages:
    split = create_splitgate_voltages_from_mV(
        mode="symmetric",
        symmetric_Vg_mV=split_Vg_mV,
        V_left_mV=0.0,
        V_right_mV=0.0,
    )
    trench = create_trenchgate_voltages_from_mV(trench_Vg_mV)
    return SplitTrenchVoltages(split_voltages=split, trench_voltages=trench)


def extract_1d_potential_x(phi_2d: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, y0_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    iy = int(np.argmin(np.abs(grid_y - y0_nm)))
    return grid_x, phi_2d[iy, :]


def run_one(
    gap_nm: float,
    trench_nm: float,
    depth_nm: float,
    grid: ElectrostaticsGrid,
    voltages: SplitTrenchVoltages,
    y0_nm: float,
    n_states: int,
) -> Result:
    split_gate = build_split_gate_structure(
        gap=gap_nm,
        gate_width_x=500.0,
        gate_length_y=80.0,
        two_deg_depth=depth_nm,
        use_dut_offset=False,
        do_describe=False,
        do_plot=False,
    )

    trench_gate = build_trench_gate_structure(
        x_length=trench_nm,
        y_width=500.0,
        x_offset=0.0,
        y_offset=0.0,
        two_deg_depth=depth_nm,
        split_shapes=split_gate.shapes,
        do_describe=False,
        do_plot=False,
        do_overlap_check=True,
    )

    structure = SplitTrenchStructure(split_gate=split_gate, trench_gate=trench_gate)
    pot_map = compute_split_trench_potential(structure=structure, voltages=voltages, grid=grid, screened=False)

    x_nm, V_eV = extract_1d_potential_x(pot_map.phi, grid.x, grid.y, y0_nm)
    energies_eV, _ = solve_schrodinger_1d(x=x_nm, V=V_eV, m_eff_kg=M_EFF_GAAS_KG, n_states=n_states)

    return Result(gap_nm=gap_nm, trench_nm=trench_nm, x_nm=x_nm, V_eV=V_eV, energies_eV=energies_eV)


def run_all(
    config_pairs: List[Tuple[float, float]],
    depth_nm: float,
    grid: ElectrostaticsGrid,
    voltages: SplitTrenchVoltages,
    y0_nm: float,
    n_states: int,
) -> List[Result]:
    results: List[Result] = []
    for i, (gap, trench) in enumerate(config_pairs, start=1):
        print(f"[{i}/{len(config_pairs)}] gap={gap} nm, trench={trench} nm")
        r = run_one(gap, trench, depth_nm, grid, voltages, y0_nm, n_states)
        print(f"  E0={r.E0_meV:.3f} meV, ΔE={r.dE_meV:.3f} meV")
        results.append(r)
    return results


def print_summary(results: List[Result], E_FERMI_EV: float) -> None:
    Ef_meV = E_FERMI_EV * 1e3
    print("\n" + "=" * 72)
    print(f"{'Gap[nm]':>8} {'Trench[nm]':>10} {'E0[meV]':>10} {'ΔE[meV]':>10} {'Modes':>8}")
    print("-" * 72)
    for r in results:
        n_modes = int(np.sum(r.energies_eV < E_FERMI_EV))
        print(f"{r.gap_nm:8.0f} {r.trench_nm:10.0f} {r.E0_meV:10.3f} {r.dE_meV:10.3f} {n_modes:8d}")
    print("-" * 72)
    print(f"Fermi energy: {Ef_meV:.3f} meV")
    print("=" * 72)


# -----------------------------
# Minimal plots (2개만)
# -----------------------------
def plot_deltaE_vs_trench(results: List[Result], out: Path | None = None) -> None:
    trenches = [r.trench_nm for r in results]
    dEs = [r.dE_meV for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(trenches, dEs, "o-", linewidth=2.0)
    ax.set_xlabel("Trench Gate Width [nm]")
    ax.set_ylabel(r"Subband Spacing $\Delta E = E_1 - E_0$ [meV]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")


def plot_overlay_Vx(results: List[Result], xlim: Tuple[float, float] | None = None, out: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        x = r.x_nm
        V = r.V_eV * 1e3
        if xlim is not None:
            m = (x >= xlim[0]) & (x <= xlim[1])
            x, V = x[m], V[m]
        ax.plot(x, V, linewidth=2.0, alpha=0.85, label=f"{r.trench_nm:.0f} nm (ΔE={r.dE_meV:.2f})")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("V(x) [meV]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")


# -----------------------------
# Main
# -----------------------------
def main() -> List[Result]:
    print_2deg_parameters()

    # configs
    GAP_NM = 100.0
    CONFIGS = [(GAP_NM, w) for w in range(10, 100, 10)]

    # physics / grid / solver
    E_FERMI_EV = 8.1e-3
    DEPTH_NM = 60.0
    Y0_NM = 0.0
    N_STATES = 10

    grid = make_uniform_grid(
        x_min=-300.0, x_max=300.0,
        y_min=-300.0, y_max=300.0,
        nx=501, ny=501,
        depth_d=DEPTH_NM,
    )
    voltages = configure_voltages(split_Vg_mV=-200.0, trench_Vg_mV=1000.0)

    results = run_all(CONFIGS, DEPTH_NM, grid, voltages, Y0_NM, N_STATES)
    print_summary(results, E_FERMI_EV)

    outdir = Path("output/subband_miniplot")
    outdir.mkdir(parents=True, exist_ok=True)

    plot_deltaE_vs_trench(results, outdir / "deltaE_vs_trench.png")
    plot_overlay_Vx(results, xlim=(-200, 200), out=outdir / "Vx_overlay.png")

    plt.show()
    return results


if __name__ == "__main__":
    main()