"""
structure_loader.py
Load bulk crystal structures from MPID (Materials Project) or POSCAR file bytes.
All functions return a pymatgen Structure in conventional standard setting.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def from_mpid(mpid: str, api_key: str) -> tuple[Structure, str]:
    """
    Fetch bulk structure from Materials Project.

    Parameters
    ----------
    mpid    : Materials Project ID, e.g. "mp-10905"
    api_key : MP API key (from https://materialsproject.org/api)

    Returns
    -------
    (structure, formula_label)
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError(
            "mp-api is required for MPID lookup.\n"
            "Install: pip install mp-api"
        )

    with MPRester(api_key) as mpr:
        s: Structure = mpr.get_structure_by_material_id(
            mpid, conventional_unit_cell=True
        )

    sga = SpacegroupAnalyzer(s, symprec=1e-2)
    std = sga.get_conventional_standard_structure()
    label = std.composition.reduced_formula
    return std, label


def from_poscar(file_bytes: bytes) -> tuple[Structure, str]:
    """
    Parse a POSCAR/CONTCAR uploaded as raw bytes.

    Parameters
    ----------
    file_bytes : Raw bytes from st.file_uploader

    Returns
    -------
    (structure, formula_label)
    """
    with tempfile.NamedTemporaryFile(suffix=".vasp", delete=False, mode="wb") as f:
        f.write(file_bytes)
        tmp_path = f.name

    s = Structure.from_file(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    sga = SpacegroupAnalyzer(s, symprec=1e-2)
    std = sga.get_conventional_standard_structure()
    label = std.composition.reduced_formula
    return std, label


def structure_info(structure: Structure) -> dict:
    """
    Extract basic info for display in the UI.

    Returns dict with keys: formula, spacegroup, a, b, c, alpha, beta, gamma, natoms
    """
    sga = SpacegroupAnalyzer(structure, symprec=1e-2)
    sg = sga.get_space_group_symbol()
    sg_num = sga.get_space_group_number()
    latt = structure.lattice
    return {
        "Formula": structure.composition.reduced_formula,
        "Spacegroup": f"{sg} (#{sg_num})",
        "a (Å)": f"{latt.a:.4f}",
        "b (Å)": f"{latt.b:.4f}",
        "c (Å)": f"{latt.c:.4f}",
        "α (°)": f"{latt.alpha:.2f}",
        "β (°)": f"{latt.beta:.2f}",
        "γ (°)": f"{latt.gamma:.2f}",
        "Atoms": len(structure),
    }
