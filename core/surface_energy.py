"""
surface_energy.py
Surface energy calculation — three methods from:
  "Leveraging Pretrained Machine Learning Models for Surface Energy Prediction
   and Wulff Construction of Intermetallic Nanoparticles"
  Jin Li, Gunnar Sly, Michael J. Janik, Penn State.

All use symmetric slab models (two identical exposed surfaces).
All return SurfaceEnergyResult. Energies in eV internally; output in both eV/Å² and J/m².

Units:
  E_*         : eV
  A           : Å²
  γ (eV/Å²)  : multiply by 16.0218 to get J/m²
  μ_i°        : eV/atom  (elemental reference: stable elemental solid)
  E_comp      : eV/formula-unit  (bulk compound total energy)
  E_form      : eV/atom  (per-atom formation energy of bulk compound)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from pymatgen.core.surface import Slab

# 1 eV/Å² = 16.0218 J/m²
EV_ANG2_TO_J_M2 = 16.0218


# ─────────────────────────────────────────────────────────────────────────────
# Shared result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurfaceEnergyResult:
    """
    Standardized result for all three surface energy methods.

    Attributes
    ----------
    miller         : (h, k, l)
    energy_ev_ang2 : γ in eV/Å²
    energy_j_m2    : γ in J/m²  (= energy_ev_ang2 × 16.0218)
    converged      : fit quality flag (always True for Methods 1/2; R²-based for Method 3)
    method         : "method1_explicit" | "method2_formation" | "method3_nlimit"
    metadata       : method-specific diagnostics
    """
    miller: tuple[int, int, int]
    energy_ev_ang2: float
    energy_j_m2: float
    converged: bool
    method: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_ev_ang2(
        cls,
        miller: tuple[int, int, int],
        energy_ev_ang2: float,
        converged: bool,
        method: str,
        metadata: dict | None = None,
    ) -> "SurfaceEnergyResult":
        return cls(
            miller=miller,
            energy_ev_ang2=energy_ev_ang2,
            energy_j_m2=energy_ev_ang2 * EV_ANG2_TO_J_M2,
            converged=converged,
            method=method,
            metadata=metadata or {},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slab_composition(slab) -> dict[str, int]:
    """Return {element_symbol: count} for atoms in slab (pymatgen Slab or Structure)."""
    from pymatgen.core import Element
    comp = slab.composition
    return {str(el): int(comp[el]) for el in comp.elements}


def _slab_surface_area(slab) -> float:
    """Return surface area (Å²) of one face — works for both Slab and Structure."""
    if hasattr(slab, "surface_area"):
        return float(slab.surface_area)
    # Fallback: |a × b| from the lattice (first two vectors span the surface plane)
    import numpy as np
    m = slab.lattice.matrix
    return float(np.linalg.norm(np.cross(m[0], m[1])))


def _miller(slab) -> tuple[int, int, int]:
    """Works for both Slab (has miller_index) and plain Structure (returns (0,0,0))."""
    if hasattr(slab, "miller_index"):
        return tuple(int(x) for x in slab.miller_index)
    return (0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — Explicit thermodynamic decomposition
#
# Paper §2.4, "Method 1":
#
#   Let the slab contain a_i atoms of element M_i (i = 1..n).
#   Let the bulk compound M_1^x1 ... M_n^xn have energy E_comp per f.u.
#   Let μ_i° = energy per atom of stable elemental solid M_i.
#
#   The slab is decomposed into (a_n / x_n) formula units of the compound
#   plus residual atoms of elements j = 1..n-1 exchanged with their
#   elemental reservoirs.
#
#   γ = 1/(2A) × [ E_slab
#                  - (a_n / x_n) × E_comp
#                  - Σ_{j=1}^{n-1} (a_j - (a_n/x_n)×x_j) × μ_j° ]
#
# "n" is the chosen normalising element (conventionally the majority or
#  last element in the compound formula).
# ─────────────────────────────────────────────────────────────────────────────

def calc_surface_energy_method1(
    slab: Slab,
    slab_energy_ev: float,
    bulk_comp_energy_ev: float,
    bulk_stoich: dict[str, float],
    elemental_refs_ev: dict[str, float],
    normalizing_element: str | None = None,
    area_ang2: float | None = None,
    miller: tuple[int, int, int] | None = None,
) -> SurfaceEnergyResult:
    """
    Method 1 — Explicit thermodynamic decomposition.

    Parameters
    ----------
    slab                 : pymatgen Slab
    slab_energy_ev       : Total energy of the relaxed slab (eV)
    bulk_comp_energy_ev  : Total energy of the bulk compound per formula unit (eV/f.u.)
                           e.g. for PdAl: E(PdAl bulk unit cell) / n_formula_units
    bulk_stoich          : Stoichiometry of the bulk compound as {element: x_i}.
                           e.g. {"Pd": 1, "Al": 1} for PdAl, {"Pd": 3, "Al": 1} for Pd3Al.
    elemental_refs_ev    : Elemental reference energies μ_i° in eV/atom.
                           {element: E_per_atom} from DFT/MLFF of stable elemental solids.
                           e.g. {"Pd": -5.18, "Al": -3.74}
    normalizing_element  : Element to use for formula-unit decomposition (default: last element
                           in bulk_stoich). All other elements are the "residual" ones.
    area_ang2            : Surface area override (Å²). Defaults to slab.surface_area.

    Returns
    -------
    SurfaceEnergyResult with method="method1_explicit"

    Formula
    -------
    Choose normalising element n (e.g. Al in PdAl).
    λ = a_n / x_n                       (number of formula units in slab)
    γ = 1/(2A) × [E_slab - λ × E_comp
                  - Σ_{i ≠ n} (a_i - λ × x_i) × μ_i°]

    Notes
    -----
    - Requires symmetric slab (both surfaces identical).
    - Works for multicomponent intermetallics regardless of stoichiometry deviation.
    - More general than Method 2 but requires individual elemental reference calculations.
    """
    # Slab composition
    slab_comp = _slab_composition(slab)
    area = area_ang2 if area_ang2 is not None else _slab_surface_area(slab)

    # Choose normalising element
    elements = list(bulk_stoich.keys())
    if normalizing_element is None:
        norm_elem = elements[-1]
    else:
        norm_elem = normalizing_element

    if norm_elem not in bulk_stoich:
        raise ValueError(f"normalizing_element '{norm_elem}' not found in bulk_stoich: {bulk_stoich}")

    x_n = bulk_stoich[norm_elem]
    a_n = slab_comp.get(norm_elem, 0)
    lam = a_n / x_n  # number of formula units in slab

    # Residual element correction: Σ_{i ≠ n} (a_i - λ × x_i) × μ_i°
    residual_correction = 0.0
    for elem in elements:
        if elem == norm_elem:
            continue
        a_i = slab_comp.get(elem, 0)
        x_i = bulk_stoich[elem]
        mu_i = elemental_refs_ev.get(elem)
        if mu_i is None:
            raise ValueError(f"Missing elemental reference energy for '{elem}' in elemental_refs_ev")
        residual_correction += (a_i - lam * x_i) * mu_i

    gamma_ev_ang2 = (slab_energy_ev - lam * bulk_comp_energy_ev - residual_correction) / (2.0 * area)

    return SurfaceEnergyResult.from_ev_ang2(
        miller=miller if miller is not None else _miller(slab),
        energy_ev_ang2=gamma_ev_ang2,
        converged=True,
        method="method1_explicit",
        metadata={
            "normalizing_element": norm_elem,
            "lambda_formula_units": lam,
            "a_n": a_n,
            "x_n": x_n,
            "slab_composition": slab_comp,
            "residual_correction_ev": residual_correction,
            "area_ang2": area,
            "slab_energy_ev": slab_energy_ev,
            "bulk_comp_energy_ev": bulk_comp_energy_ev,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — Bulk-like reference based on formation energy
#
# Paper §2.4, "Method 2":
#
#   E_ref = Σ_i a_i × μ_i°  +  N × E_form
#
#   where N = Σ_i a_i (total atoms in slab),
#         E_form = per-atom formation energy of the bulk compound (eV/atom).
#
#   γ = (E_slab - E_ref) / (2A)
#
# Applicable even when the slab termination deviates from bulk stoichiometry.
# ─────────────────────────────────────────────────────────────────────────────

def calc_surface_energy_method2(
    slab: Slab,
    slab_energy_ev: float,
    elemental_refs_ev: dict[str, float],
    bulk_formation_energy_per_atom_ev: float,
    area_ang2: float | None = None,
    miller: tuple[int, int, int] | None = None,
) -> SurfaceEnergyResult:
    """
    Method 2 — Formation energy reference.

    Parameters
    ----------
    slab                              : pymatgen Slab
    slab_energy_ev                    : Total energy of the relaxed slab (eV)
    elemental_refs_ev                 : Elemental reference energies μ_i° (eV/atom).
                                        {element: E_per_atom} for each element in slab.
    bulk_formation_energy_per_atom_ev : Per-atom formation energy of the bulk compound (eV/atom).
                                        E_form = (E_bulk_compound - Σ x_i × μ_i°) / N_atoms_per_fu
                                        Negative for stable compounds.
    area_ang2                         : Surface area override (Å²). Defaults to slab.surface_area.

    Returns
    -------
    SurfaceEnergyResult with method="method2_formation"

    Formula
    -------
    E_ref = Σ_i (a_i × μ_i°)  +  N × E_form
    γ     = (E_slab − E_ref) / (2A)

    Notes
    -----
    - Handles off-stoichiometric slab terminations naturally.
    - Preserves thermodynamic consistency with the bulk phase.
    - E_form should be computed as: (E_compound_per_fu - Σ x_i×μ_i°) / N_atoms_per_fu
      using the SAME calculator (DFT or MLFF) for all energies.
    """
    slab_comp = _slab_composition(slab)
    area = area_ang2 if area_ang2 is not None else _slab_surface_area(slab)
    n_total = sum(slab_comp.values())

    # Reference energy: elemental parts + formation energy correction
    e_elemental = 0.0
    for elem, count in slab_comp.items():
        mu = elemental_refs_ev.get(elem)
        if mu is None:
            raise ValueError(f"Missing elemental reference for '{elem}' in elemental_refs_ev")
        e_elemental += count * mu

    e_ref = e_elemental + n_total * bulk_formation_energy_per_atom_ev
    gamma_ev_ang2 = (slab_energy_ev - e_ref) / (2.0 * area)

    return SurfaceEnergyResult.from_ev_ang2(
        miller=miller if miller is not None else _miller(slab),
        energy_ev_ang2=gamma_ev_ang2,
        converged=True,
        method="method2_formation",
        metadata={
            "slab_composition": slab_comp,
            "n_total_atoms": n_total,
            "e_elemental_ev": e_elemental,
            "e_ref_ev": e_ref,
            "bulk_formation_energy_per_atom_ev": bulk_formation_energy_per_atom_ev,
            "area_ang2": area,
            "slab_energy_ev": slab_energy_ev,
        },
    )


def calc_bulk_formation_energy_per_atom(
    bulk_energy_per_fu_ev: float,
    bulk_stoich: dict[str, float],
    elemental_refs_ev: dict[str, float],
) -> float:
    """
    Compute per-atom formation energy of the bulk compound.

    E_form = (E_comp_per_fu - Σ x_i × μ_i°) / N_atoms_per_fu

    Parameters
    ----------
    bulk_energy_per_fu_ev : Total bulk energy per formula unit (eV/f.u.)
    bulk_stoich           : {element: x_i} stoichiometry
    elemental_refs_ev     : {element: μ_i° in eV/atom}

    Returns
    -------
    E_form in eV/atom (negative for stable compounds)
    """
    n_atoms_per_fu = sum(bulk_stoich.values())
    e_ref_per_fu = sum(bulk_stoich[el] * elemental_refs_ev[el] for el in bulk_stoich)
    return (bulk_energy_per_fu_ev - e_ref_per_fu) / n_atoms_per_fu


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — Unified linear model / N-limit
#
# Paper §2.4, "Method 3":
#
# For any configuration α (bulk or slab) with a_{i,α} atoms of element M_i,
# total atom count N_α, and surface area A_α (= 0 for bulk):
#
#   E_α = Σ_i a_{i,α}×μ_i°  +  N_α×E_form  +  2γ×A_α  +  ε_α
#
# This is linear in the unknowns (μ_1°,...,μ_n°, E_form, γ).
# Collecting all bulk and slab configurations → overdetermined system
# solved by least-squares (numpy.linalg.lstsq).
#
# For the simplified case (single termination, thickness series):
#   E_slab(N) = E_ref(N) + 2γA + ε(N)
#   with E_ref(N) = Σ_i a_i(N)×μ_i° + N×E_form
#   → γ = intercept of linear fit of [E_slab(N) - E_ref(N)] vs N
#
# Two modes:
#   mode="global"    : Full least-squares including all bulk + slab configs
#                      (determines μ_i°, E_form, γ simultaneously)
#   mode="n_limit"   : Simplified: use known μ_i° and E_form, extrapolate γ
#                      from thickness series via [E_slab - E_ref] vs N fit
# ─────────────────────────────────────────────────────────────────────────────

def calc_surface_energy_method3_nlimit(
    slabs: list[Slab],
    slab_energies_ev: list[float],
    elemental_refs_ev: dict[str, float],
    bulk_formation_energy_per_atom_ev: float,
    area_ang2: float | None = None,
    min_r2: float = 0.999,
) -> SurfaceEnergyResult:
    """
    Method 3 (N-limit) — Extrapolation from thickness series.

    Uses known elemental references and formation energy (from Methods 1/2 or
    independent calculation). Plots [E_slab(N) - E_ref(N)] vs N and extracts
    γ from the intercept at N→∞.

    According to the paper, linearity of [E_slab(N) - E_ref(N)] vs N is approached
    only asymptotically (thick-slab regime). This method therefore requires slabs
    in the thick, converged regime.

    Parameters
    ----------
    slabs                             : List of Slabs, same termination, increasing thickness.
                                        Must be in the asymptotic (converged) regime.
                                        Recommended: ≥ 5 slabs, all beyond oscillatory regime.
    slab_energies_ev                  : Total energies (eV), same order as slabs.
    elemental_refs_ev                 : {element: μ_i° in eV/atom}
    bulk_formation_energy_per_atom_ev : E_form (eV/atom), same calculator as slab energies.
    area_ang2                         : Surface area override (Å²). Defaults to slabs[0].surface_area.
    min_r2                            : R² threshold to flag convergence (default 0.999).

    Returns
    -------
    SurfaceEnergyResult with method="method3_nlimit"
    metadata includes: R², intercept (= 2γA), slope (finite-size coefficient),
                       [E_slab - E_ref] values, N values, convergence_flag.

    Formula
    -------
    E_ref(N) = Σ_i a_i(N) × μ_i°  +  N × E_form
    δE(N)    = E_slab(N) - E_ref(N)          [should be linear in N asymptotically]
    γ        = intercept of fit δE(N) = b + c×N   divided by (2A)
             = b / (2A)
    """
    if len(slabs) < 3:
        raise ValueError("Method 3 N-limit requires at least 3 slabs in the thick-slab regime.")

    area = area_ang2 if area_ang2 is not None else _slab_surface_area(slabs[0])

    n_atoms_list = []
    delta_e_list = []

    for slab, e_slab in zip(slabs, slab_energies_ev):
        comp = _slab_composition(slab)
        n_total = sum(comp.values())

        # E_ref(N) = Σ a_i × μ_i° + N × E_form
        e_elemental = sum(comp.get(el, 0) * elemental_refs_ev[el] for el in comp)
        e_ref = e_elemental + n_total * bulk_formation_energy_per_atom_ev

        delta_e = e_slab - e_ref  # should → 2γA + const as N → ∞
        n_atoms_list.append(float(n_total))
        delta_e_list.append(float(delta_e))

    n_arr = np.array(n_atoms_list)
    de_arr = np.array(delta_e_list)

    # Fit δE(N) = b + c × N  (intercept b = 2γA)
    # Use polyfit with deg=1: coefficients [c, b]
    coeffs = np.polyfit(n_arr, de_arr, deg=1)
    c_slope, b_intercept = coeffs

    # R² of the fit
    de_pred = np.polyval(coeffs, n_arr)
    ss_res = np.sum((de_arr - de_pred) ** 2)
    ss_tot = np.sum((de_arr - np.mean(de_arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    gamma_ev_ang2 = b_intercept / (2.0 * area)
    converged = (r2 >= min_r2)

    return SurfaceEnergyResult.from_ev_ang2(
        miller=_miller(slabs[0]),
        energy_ev_ang2=gamma_ev_ang2,
        converged=converged,
        method="method3_nlimit",
        metadata={
            "r2": float(r2),
            "intercept_2gamma_A_ev": float(b_intercept),
            "slope_finite_size_ev_per_atom": float(c_slope),
            "area_ang2": area,
            "n_slabs": len(slabs),
            "n_atoms_list": n_atoms_list,
            "delta_e_list": delta_e_list,
            "bulk_formation_energy_per_atom_ev": bulk_formation_energy_per_atom_ev,
            "convergence_flag": converged,
            "min_r2_threshold": min_r2,
        },
    )


def calc_surface_energy_method3_global(
    configs: list[dict],
    n_elements: int,
    element_order: list[str],
    area_ang2: float,
) -> dict:
    """
    Method 3 (global) — Simultaneous least-squares over all bulk + slab configs.

    Solves the overdetermined linear system:
        E_α = Σ_i a_{i,α}×μ_i°  +  N_α×E_form  +  2γ×A_α

    for unknowns: (μ_1°, ..., μ_n°, E_form, γ).
    All bulk configs have A_α = 0; slab configs have A_α = area.

    Parameters
    ----------
    configs       : List of configuration dicts, each with:
                      "composition"  : {element: count}
                      "n_atoms"      : int (= sum of composition values)
                      "energy_ev"    : float (total DFT/MLFF energy)
                      "area_ang2"    : float (0.0 for bulk, surface area for slabs)
                      "miller"       : tuple or None (None for bulk)
                      "label"        : str (for diagnostics)
    n_elements    : Number of distinct elements in the system
    element_order : List of element symbols in fixed order (determines column order in matrix)
    area_ang2     : Surface area for slab configs (Å²). Used to normalize γ.

    Returns
    -------
    dict with keys:
        "mu_refs"      : {element: μ_i° in eV/atom}  (fitted elemental references)
        "e_form"       : float  (fitted per-atom formation energy, eV/atom)
        "gamma_ev_ang2": dict {(h,k,l): γ in eV/Å²} — one γ per unique miller index
        "gamma_j_m2"   : dict {(h,k,l): γ in J/m²}
        "residuals_ev" : list of per-config residuals
        "rank"         : matrix rank (for conditioning diagnostics)
        "rcond"        : condition number proxy

    Notes
    -----
    - Requires sufficient bulk AND slab diversity for the system to be well-conditioned.
    - Each unique slab termination (miller index + shift) introduces one γ unknown.
    - For a single termination (all slabs same facet), reduces to one γ.
    - Recommended: include ≥ 3 bulk supercell sizes and ≥ 3 slab thicknesses per facet.
    """
    # Identify unique miller indices among slab configs
    unique_millers = []
    miller_to_idx = {}
    for cfg in configs:
        m = cfg.get("miller")
        if m is not None and m not in miller_to_idx:
            miller_to_idx[m] = len(unique_millers)
            unique_millers.append(m)

    # Number of unknowns: n_elements (μ_i°) + 1 (E_form) + n_facets (γ per facet)
    n_facets = len(unique_millers)
    n_unknowns = n_elements + 1 + n_facets
    elem_idx = {el: i for i, el in enumerate(element_order)}

    # Build matrix A and vector b for: A @ x = b
    A_rows = []
    b_vals = []
    config_labels = []

    for cfg in configs:
        comp = cfg["composition"]
        n_total = cfg["n_atoms"]
        e = cfg["energy_ev"]
        a_fac = cfg.get("area_ang2", 0.0)
        miller = cfg.get("miller")

        row = np.zeros(n_unknowns)

        # Elemental reference columns: a_{i,α}
        for el, count in comp.items():
            if el in elem_idx:
                row[elem_idx[el]] = count

        # E_form column: N_α
        row[n_elements] = float(n_total)

        # γ column(s): 2 × A_α for the appropriate facet
        if miller is not None and a_fac > 0:
            facet_col = n_elements + 1 + miller_to_idx[miller]
            row[facet_col] = 2.0 * a_fac

        A_rows.append(row)
        b_vals.append(float(e))
        config_labels.append(cfg.get("label", "?"))

    A = np.array(A_rows)
    b = np.array(b_vals)

    # Solve least-squares
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    # Extract results
    mu_refs = {element_order[i]: float(x[i]) for i in range(n_elements)}
    e_form = float(x[n_elements])
    gamma_by_miller = {}
    for m, idx in miller_to_idx.items():
        gamma_ev = float(x[n_elements + 1 + idx]) / (2.0 * area_ang2)
        gamma_by_miller[m] = gamma_ev

    # Per-config residuals
    b_pred = A @ x
    per_config_residuals = (b - b_pred).tolist()

    rcond_proxy = float(sv[0] / sv[-1]) if len(sv) > 1 and sv[-1] > 0 else float("inf")

    return {
        "mu_refs": mu_refs,
        "e_form": e_form,
        "gamma_ev_ang2": gamma_by_miller,
        "gamma_j_m2": {m: g * EV_ANG2_TO_J_M2 for m, g in gamma_by_miller.items()},
        "residuals_ev": per_config_residuals,
        "config_labels": config_labels,
        "rank": int(rank),
        "rcond_proxy": rcond_proxy,
        "unique_millers": unique_millers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convergence diagnostics (Method 3)
# ─────────────────────────────────────────────────────────────────────────────

def convergence_plot_data(
    slabs,  # list[Slab] | None
    slab_energies_ev: list[float],
    elemental_refs_ev: dict[str, float],
    bulk_formation_energy_per_atom_ev: float,
    area_ang2: float | None = None,
    _n_atoms_override: list[int] | None = None,   # bypass Slab objects
) -> dict:
    """
    Compute [E_slab(N) - E_ref(N)] / (2A) vs number of layers (or N atoms)
    for convergence visualization — Fig 1 style from the paper.

    Returns dict with:
        "n_atoms"         : list of atom counts
        "gamma_raw_ev_ang2": list of γ(N) = [E_slab - E_ref] / (2A) per thickness
        "gamma_raw_j_m2"  : same in J/m²
    """
    if slabs is not None:
        area = area_ang2 if area_ang2 is not None else _slab_surface_area(slabs[0])
    else:
        area = area_ang2 or 15.0

    n_atoms_list = []
    gamma_list = []

    if _n_atoms_override is not None:
        # Fast path: N and E provided directly (no Slab objects needed)
        for n_total, e_slab in zip(_n_atoms_override, slab_energies_ev):
            # Approximate elemental correction using bulk stoichiometry fractions
            # (caller has already subtracted E_ref in the app — here we compute δE/2A)
            # Use simple N × E_form approach since we don't have per-element counts
            e_ref = n_total * bulk_formation_energy_per_atom_ev
            gamma_n = (e_slab - e_ref) / (2.0 * area)
            n_atoms_list.append(n_total)
            gamma_list.append(gamma_n)
    else:
        for slab, e_slab in zip(slabs, slab_energies_ev):
            comp = _slab_composition(slab)
            n_total = sum(comp.values())
            e_elemental = sum(comp.get(el, 0) * elemental_refs_ev[el] for el in comp)
            e_ref = e_elemental + n_total * bulk_formation_energy_per_atom_ev
            gamma_n = (e_slab - e_ref) / (2.0 * area)
            n_atoms_list.append(n_total)
            gamma_list.append(gamma_n)

    return {
        "n_atoms": n_atoms_list,
        "gamma_raw_ev_ang2": gamma_list,
        "gamma_raw_j_m2": [g * EV_ANG2_TO_J_M2 for g in gamma_list],
    }


def calc_surface_energy_method3_direct(
    n_atoms_list: list[int],
    slab_energies_ev: list[float],
    area_ang2: float,
    miller: tuple = (0, 0, 0),
    min_r2: float = 0.99,
) -> SurfaceEnergyResult:
    """
    Method 3 — Reference-free direct linear fit.

    Fits E_slab = a * N + b directly (no external elemental or formation energies).
        slope a  → bulk energy per atom (self-determined)
        intercept b → 2 * A * γ

    Returns SurfaceEnergyResult with:
        method = "method3_direct"
        metadata["slope_ev_atom"] = a  (bulk energy per atom from fit)
        metadata["r2"] = fit quality
    """
    n_arr = np.array(n_atoms_list, dtype=float)
    e_arr = np.array(slab_energies_ev, dtype=float)

    if len(n_arr) < 2:
        raise ValueError("Need at least 2 data points for linear fit.")

    coeffs = np.polyfit(n_arr, e_arr, deg=1)
    slope, intercept = coeffs  # E = slope*N + intercept

    gamma_ev_ang2 = intercept / (2.0 * area_ang2)

    e_pred = np.polyval(coeffs, n_arr)
    ss_res = float(np.sum((e_arr - e_pred) ** 2))
    ss_tot = float(np.sum((e_arr - np.mean(e_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    return SurfaceEnergyResult.from_ev_ang2(
        miller=miller,
        energy_ev_ang2=gamma_ev_ang2,
        converged=(r2 >= min_r2),
        method="method3_direct",
        metadata={
            "r2": r2,
            "slope_ev_atom": float(slope),
            "intercept": float(intercept),
            "area_ang2": area_ang2,
            "n_points": len(n_atoms_list),
        },
    )


def convergence_plot_data_direct(
    n_atoms_list: list[int],
    slab_energies_ev: list[float],
    area_ang2: float,
) -> dict:
    """
    Convergence plot data for direct method 3 (no references).
    Returns per-point γ assuming bulk energy from a running linear fit.
    Also returns the final fitted line for overlay.
    """
    n_arr = np.array(n_atoms_list, dtype=float)
    e_arr = np.array(slab_energies_ev, dtype=float)

    # Per-point γ: for each point, use all points up to that one to fit
    gamma_per_point = []
    for i in range(2, len(n_arr) + 1):
        coeffs_i = np.polyfit(n_arr[:i], e_arr[:i], deg=1)
        gamma_per_point.append(float(coeffs_i[1] / (2.0 * area_ang2)))

    # Full fit line
    coeffs_full = np.polyfit(n_arr, e_arr, deg=1)
    n_fine = np.linspace(float(n_arr.min()), float(n_arr.max()), 200).tolist()
    e_fit = np.polyval(coeffs_full, np.array(n_fine)).tolist()

    return {
        "n_atoms": n_atoms_list,
        "e_slab_ev": list(slab_energies_ev),
        "gamma_running_j_m2": [g * EV_ANG2_TO_J_M2 for g in gamma_per_point],
        "n_fit": n_fine,
        "e_fit": e_fit,
        "slope": float(coeffs_full[0]),
        "intercept": float(coeffs_full[1]),
    }
