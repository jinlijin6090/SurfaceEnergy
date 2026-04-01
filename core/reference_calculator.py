"""
reference_calculator.py
Fetch elemental bulk structures from Materials Project and compute formation energies.
"""

from __future__ import annotations

from pymatgen.core import Structure


def fetch_elemental_structures(
    elements: list[str],
    api_key: str,
) -> dict[str, dict]:
    """
    Fetch the lowest-energy stable elemental bulk structure for each element from MP.

    Returns
    -------
    dict mapping element symbol → {
        "structure"   : pymatgen Structure,
        "mpid"        : str,
        "formula"     : str,
        "dft_energy_per_atom" : float (eV/atom from MP GGA/GGA+U),
    }
    """
    from mp_api.client import MPRester

    results = {}
    with MPRester(api_key) as mpr:
        for el in elements:
            docs = mpr.summary.search(
                elements=[el],
                num_elements=1,
                is_stable=True,
                fields=["material_id", "structure", "energy_per_atom", "formula_pretty"],
            )
            if not docs:
                # Fallback: search without stability filter
                docs = mpr.summary.search(
                    elements=[el],
                    num_elements=1,
                    fields=["material_id", "structure", "energy_per_atom", "formula_pretty"],
                )
            if docs:
                best = min(docs, key=lambda d: d.energy_per_atom)
                results[el] = {
                    "structure": best.structure,
                    "mpid": str(best.material_id),
                    "formula": best.formula_pretty,
                    "dft_energy_per_atom": best.energy_per_atom,
                }
    return results


def calc_formation_energy_per_atom(
    compound_energy_per_atom_ev: float,
    compound_composition: dict[str, float],
    elemental_energies_ev: dict[str, float],
) -> float:
    """
    Compute formation energy per atom of a compound.

    E_form/atom = E_compound/atom - Σ (x_i * μ_i°)
    where x_i = atom fraction of element i.

    Parameters
    ----------
    compound_energy_per_atom_ev : total DFT/MLFF energy divided by N_atoms
    compound_composition        : {element: stoich_count}, e.g. {"Pd": 1, "Al": 1}
    elemental_energies_ev       : {element: eV/atom}

    Returns
    -------
    float : formation energy per atom (eV/atom), negative for stable compounds
    """
    total = sum(compound_composition.values())
    e_elem_weighted = sum(
        (compound_composition[el] / total) * elemental_energies_ev[el]
        for el in compound_composition
        if el in elemental_energies_ev
    )
    return compound_energy_per_atom_ev - e_elem_weighted
