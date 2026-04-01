"""
surface_generator.py
Slab generation using pymatgen SlabGenerator.
Adapted from collect_local.py — single-threaded for use in Streamlit.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import plotly.graph_objects as go
from ase.constraints import FixAtoms
from pymatgen.core import Structure
from pymatgen.core.surface import (
    Slab,
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pymatgen.io.ase import AseAtomsAdaptor

# CPK-like color map (element symbol -> hex color)
_CPK_COLORS: dict[str, str] = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "S": "#FFFF30", "Cl": "#1FF01F", "Fe": "#E06633",
    "Cu": "#C88033", "Pd": "#006985", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Ag": "#C0C0C0", "Al": "#BFA6A6", "Ti": "#BFC2C7", "Ni": "#50D050",
    "Co": "#F090A0", "Cr": "#8A99C7", "Mn": "#9C7AC7", "Zn": "#7D80B0",
    "Sb": "#9E63B5", "In": "#A67573", "Sn": "#668080", "Mo": "#54B5B5",
    "W": "#2194D6", "Ir": "#175487", "Ru": "#248F8F", "Rh": "#0A7D8C",
    "Re": "#267DAB", "Os": "#266696", "Si": "#F0C8A0", "Ge": "#668F8F",
    "As": "#BD80E3",
}
_DEFAULT_COLOR = "#BBBBBB"


def _element_color(symbol: str) -> str:
    return _CPK_COLORS.get(symbol, _DEFAULT_COLOR)


# Covalent radii in Angstrom (rough values for marker sizing)
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05,
    "Cl": 1.02, "Fe": 1.32, "Cu": 1.32, "Pd": 1.39, "Pt": 1.36, "Au": 1.36,
    "Ag": 1.45, "Al": 1.21, "Ti": 1.60, "Ni": 1.24, "Co": 1.26, "Cr": 1.39,
    "Mn": 1.61, "Zn": 1.22, "Si": 1.11, "Ge": 1.20, "Sn": 1.39, "Mo": 1.54,
    "W": 1.62, "Ir": 1.41, "Ru": 1.46, "Rh": 1.42, "Pt": 1.36, "Sb": 1.39,
}
_DEFAULT_RADIUS = 1.20


def enumerate_miller_indices(
    structure: Structure, max_index: int = 3
) -> list[tuple[int, int, int]]:
    """Return symmetrically distinct Miller indices up to max_index."""
    return list(
        get_symmetrically_distinct_miller_indices(structure, max_index=max_index)
    )


def constrain_slab_atoms(atoms, z_cutoff: float = 4.0):
    """Fix middle atoms; allow top/bottom z_cutoff Å to relax."""
    atoms = atoms.copy()
    spz = np.asarray(atoms.get_scaled_positions())[:, 2]
    cell_h = np.linalg.norm(atoms.cell[2])
    max_z = float(np.max(spz))
    min_z = float(np.min(spz))
    upper = max_z - z_cutoff / cell_h
    lower = min_z + z_cutoff / cell_h
    mask = (spz >= lower) & (spz <= upper)
    atoms.set_constraint(FixAtoms(mask=mask))
    return atoms


def generate_slabs(
    structure: Structure,
    miller_indices: list[tuple[int, int, int]],
    min_slab_size: float = 15.0,
    min_vacuum_size: float = 15.0,
    z_cutoff: float = 4.0,
    primitive: bool = True,
    center_slab: bool = True,
    in_unit_planes: bool = False,
    tol_sym: float = 0.10,
    tol_asym: float = 0.05,
    progress_callback=None,
) -> list[dict]:
    """
    Generate slabs for each Miller index.

    Returns list of dicts with keys:
        miller   : (h, k, l) tuple
        shift    : termination shift value
        index    : termination index
        label    : human-readable string for UI dropdown
        slab     : pymatgen Slab
        symmetric: bool (True if slab was symmetrized)
        atoms    : ASE Atoms with FixAtoms constraints applied
    """
    adaptor = AseAtomsAdaptor()
    results = []
    total = len(miller_indices)

    for i, hkl in enumerate(miller_indices):
        if progress_callback:
            progress_callback(i, total, hkl)

        # Probe whether symmetric terminations exist
        try:
            sg_probe = SlabGenerator(
                initial_structure=structure,
                miller_index=hkl,
                min_slab_size=min_slab_size,
                min_vacuum_size=min_vacuum_size,
                primitive=primitive,
                center_slab=center_slab,
                in_unit_planes=in_unit_planes,
            )
            probe_slabs = sg_probe.get_slabs(tol=tol_sym, symmetrize=True)
            is_sym = len(probe_slabs) > 0
        except Exception:
            is_sym = False

        tol = tol_sym if is_sym else tol_asym

        try:
            sg = SlabGenerator(
                initial_structure=structure,
                miller_index=hkl,
                min_slab_size=min_slab_size,
                min_vacuum_size=min_vacuum_size,
                primitive=primitive,
                center_slab=center_slab,
                in_unit_planes=in_unit_planes,
            )
            slabs = sg.get_slabs(tol=tol, symmetrize=is_sym)
        except Exception as e:
            # Record failure as empty entry
            results.append({
                "miller": hkl,
                "shift": 0.0,
                "index": 0,
                "label": f"({hkl[0]} {hkl[1]} {hkl[2]}) — FAILED: {e}",
                "slab": None,
                "symmetric": False,
                "atoms": None,
                "error": str(e),
            })
            continue

        for j, slab in enumerate(slabs):
            shift_val = getattr(slab, "shift", 0.0)
            sym_tag = "sym" if is_sym else "asym"
            label = f"({hkl[0]} {hkl[1]} {hkl[2]})  shift={shift_val:.4f}  [{sym_tag}]  term-{j}"

            try:
                ase_atoms = adaptor.get_atoms(slab)
                ase_atoms = constrain_slab_atoms(ase_atoms, z_cutoff=z_cutoff)
            except Exception:
                ase_atoms = None

            results.append({
                "miller": hkl,
                "shift": shift_val,
                "index": j,
                "label": label,
                "slab": slab,
                "symmetric": is_sym,
                "atoms": ase_atoms,
                "error": None,
            })

    return results


def parse_miller_string(text: str) -> list[tuple[int, int, int]]:
    """
    Parse user-typed Miller indices from text.
    Accepts formats like: "1,1,1  0,0,1  1,1,0" or "111 001 110"
    Returns list of (h,k,l) tuples.
    """
    import re

    results = []
    # Try comma-separated triplets: each component is 1-2 digits  e.g. "1,1,1" or "1 1 0"
    triplets = re.findall(r"-?\d{1,2}[,\s]+-?\d{1,2}[,\s]+-?\d{1,2}", text)
    for t in triplets:
        parts = re.findall(r"-?\d+", t)
        if len(parts) == 3:
            results.append((int(parts[0]), int(parts[1]), int(parts[2])))

    if not results:
        # Try bare 3-char tokens like "111" → (1,1,1), "1-10" not supported here
        tokens = text.split()
        for tok in tokens:
            tok = tok.strip("()[]")
            if len(tok) == 3 and tok.lstrip("-").isdigit():
                results.append((int(tok[0]), int(tok[1]), int(tok[2])))

    # Deduplicate
    seen = set()
    unique = []
    for hkl in results:
        if hkl not in seen:
            unique.append(hkl)
            seen.add(hkl)
    return unique


def slab_to_plotly_traces(slab_dict: dict) -> list[go.Scatter3d]:
    """
    Convert a slab entry (from generate_slabs) to plotly Scatter3d traces.
    One trace per element, colored by CPK scheme, sized by covalent radius.
    Fixed atoms shown as slightly smaller and with a different symbol.

    Returns list of go.Scatter3d traces ready for go.Figure(data=traces).
    """
    if slab_dict.get("atoms") is None and slab_dict.get("slab") is None:
        return []

    # Prefer ASE Atoms (has constraints); fallback to pymatgen Slab
    atoms = slab_dict.get("atoms")
    if atoms is not None:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        # Fixed atom mask
        fixed_mask = np.zeros(len(atoms), dtype=bool)
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                fixed_mask[c.index] = True
    else:
        slab = slab_dict["slab"]
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor().get_atoms(slab)
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        fixed_mask = np.zeros(len(atoms), dtype=bool)

    # Group by element
    element_groups: dict[str, dict] = {}
    for idx, (sym, pos, fixed) in enumerate(zip(symbols, positions, fixed_mask)):
        if sym not in element_groups:
            element_groups[sym] = {"x": [], "y": [], "z": [], "fixed": [], "idx": []}
        element_groups[sym]["x"].append(pos[0])
        element_groups[sym]["y"].append(pos[1])
        element_groups[sym]["z"].append(pos[2])
        element_groups[sym]["fixed"].append(fixed)
        element_groups[sym]["idx"].append(idx)

    traces = []
    for sym, data in element_groups.items():
        color = _element_color(sym)
        radius = _COVALENT_RADII.get(sym, _DEFAULT_RADIUS)
        # Size: fixed atoms smaller
        sizes = [int(radius * 12) if not f else int(radius * 8)
                 for f in data["fixed"]]
        hover_texts = [
            f"{sym} #{i}<br>({'fixed' if f else 'free'})<br>"
            f"x={x:.3f} y={y:.3f} z={z:.3f}"
            for i, f, x, y, z in zip(
                data["idx"], data["fixed"],
                data["x"], data["y"], data["z"]
            )
        ]
        traces.append(go.Scatter3d(
            x=data["x"], y=data["y"], z=data["z"],
            mode="markers",
            name=sym,
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.9,
                line=dict(width=0.5, color="black"),
            ),
        ))
    return traces


def structure_to_plotly_traces(structure: Structure) -> list[go.Scatter3d]:
    """
    Convert a pymatgen bulk Structure to plotly traces for preview.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor().get_atoms(structure)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    element_groups: dict[str, dict] = {}
    for sym, pos in zip(symbols, positions):
        if sym not in element_groups:
            element_groups[sym] = {"x": [], "y": [], "z": []}
        element_groups[sym]["x"].append(pos[0])
        element_groups[sym]["y"].append(pos[1])
        element_groups[sym]["z"].append(pos[2])

    traces = []
    for sym, data in element_groups.items():
        color = _element_color(sym)
        radius = _COVALENT_RADII.get(sym, _DEFAULT_RADIUS)
        traces.append(go.Scatter3d(
            x=data["x"], y=data["y"], z=data["z"],
            mode="markers",
            name=sym,
            marker=dict(
                size=int(radius * 14),
                color=color,
                opacity=0.9,
                line=dict(width=0.5, color="black"),
            ),
        ))
    return traces
