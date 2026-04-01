"""
optimizer.py
Structure optimization backends: MLFF (fairchem 2.x / OCP checkpoint) and DFT (VASP file generation).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor


# ─────────────────────────────────────────────────────────────────────────────
# MLFF relaxation using fairchem 2.x
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_UMA_MODELS = [
    "uma-s-1",
    "uma-s-1p1",
    "uma-m-1p1",
    "esen-sm-conserving-all-omol",
    "esen-sm-direct-all-omol",
    "esen-sm-conserving-all-oc25",
    "esen-md-direct-all-oc25",
]

# Module-level caches: avoids reloading model weights and re-building calculators
_PREDICTOR_CACHE: dict[str, object] = {}
_CALCULATOR_CACHE: dict[tuple, object] = {}  # key: (model_name, task_name)


def _get_predictor(model_name: str):
    """Load (or return cached) fairchem predict unit for the given model."""
    if model_name not in _PREDICTOR_CACHE:
        from fairchem.core import pretrained_mlip
        _PREDICTOR_CACHE[model_name] = pretrained_mlip.get_predict_unit(model_name)
    return _PREDICTOR_CACHE[model_name]


def _get_calculator(model_name: str, task_name: str):
    """Return a cached FAIRChemCalculator for (model_name, task_name)."""
    key = (model_name, task_name)
    if key not in _CALCULATOR_CACHE:
        from fairchem.core.units.mlip_unit import FAIRChemCalculator
        _CALCULATOR_CACHE[key] = FAIRChemCalculator(
            _get_predictor(model_name), task_name=task_name
        )
    return _CALCULATOR_CACHE[key]


def relax_mlff(
    slab_dict: dict,
    model_name: str = "uma-s-1",
    fmax: float = 0.05,
    steps: int = 200,
    task_name: str = "oc20",
    progress_callback: Callable[[int, float], None] | None = None,
) -> dict:
    """
    Relax a slab using a fairchem 2.x UMA model.

    Parameters
    ----------
    slab_dict         : Entry from surface_generator.generate_slabs()
    model_name        : UMA model name (see AVAILABLE_UMA_MODELS)
    fmax              : Force convergence criterion in eV/Å
    steps             : Maximum optimization steps
    task_name         : fairchem task name ("oc20", "omol", etc.)
    progress_callback : Optional fn(step, fmax_current) called each step

    Returns
    -------
    dict with keys:
        label         : same as input slab_dict['label']
        miller        : (h,k,l)
        converged     : bool
        energy_ev     : final energy in eV
        fmax_final    : final max force in eV/Å
        atoms         : relaxed ASE Atoms object (with constraints)
        slab          : relaxed pymatgen Slab
        steps_taken   : number of steps
    """
    try:
        _get_calculator(model_name, task_name)  # ensure importable early
    except ImportError:
        raise ImportError(
            "fairchem-core >= 2.x is required for MLFF relaxation.\n"
            "Activate: conda activate fairchem_UMA_NEW"
        )

    from ase.optimize import LBFGS

    atoms = slab_dict.get("atoms")
    if atoms is None:
        slab = slab_dict.get("slab")
        if slab is None:
            raise ValueError(f"slab_dict for {slab_dict['label']} has no atoms or slab.")
        atoms = AseAtomsAdaptor().get_atoms(slab)

    atoms = atoms.copy()

    # Reuse cached calculator — no model reload between slabs
    atoms.calc = _get_calculator(model_name, task_name)

    # Track steps
    step_counter = [0]
    fmax_log = [None]

    def obs():
        step_counter[0] += 1
        forces = atoms.get_forces()
        fmax_now = float((forces ** 2).sum(axis=1).max() ** 0.5)
        fmax_log[0] = fmax_now
        if progress_callback:
            progress_callback(step_counter[0], fmax_now)

    opt = LBFGS(atoms, logfile=None)
    opt.attach(obs, interval=1)

    try:
        converged = opt.run(fmax=fmax, steps=steps)
    except Exception as e:
        return {
            "label": slab_dict["label"],
            "miller": slab_dict["miller"],
            "converged": False,
            "energy_ev": None,
            "fmax_final": fmax_log[0],
            "atoms": atoms,
            "slab": None,
            "steps_taken": step_counter[0],
            "error": str(e),
        }

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float((forces ** 2).sum(axis=1).max() ** 0.5)

    # Convert back to pymatgen Slab
    try:
        relaxed_slab = AseAtomsAdaptor().get_structure(atoms)
    except Exception:
        relaxed_slab = None

    return {
        "label": slab_dict["label"],
        "miller": slab_dict["miller"],
        "converged": bool(converged),
        "energy_ev": energy,
        "fmax_final": fmax_final,
        "atoms": atoms,
        "slab": relaxed_slab,
        "steps_taken": step_counter[0],
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MLFF relaxation using a local OCP checkpoint (fairchem 1.x OCPCalculator)
# Runs in a subprocess under fairchem_old_OCP conda env.
# ─────────────────────────────────────────────────────────────────────────────

#: Path to the conda environment that has OCPCalculator installed.
OCP_CONDA_ENV = "fairchem_old_OCP"


def relax_ocp_checkpoint(
    slab_dict: dict,
    checkpoint_path: str,
    fmax: float = 0.05,
    steps: int = 200,
    cpu: bool = True,
    conda_env: str = OCP_CONDA_ENV,
) -> dict:
    """
    Relax a slab using a local OCP checkpoint (GNOc / GemNet-OC / etc.).

    Runs OCPCalculator in a subprocess under ``conda_env`` so it can coexist
    with the fairchem 2.x Streamlit environment.

    Parameters
    ----------
    slab_dict       : Entry from surface_generator.generate_slabs()
    checkpoint_path : Absolute path to the .pt checkpoint file
    fmax            : Force convergence criterion (eV/Å)
    steps           : Max optimisation steps
    cpu             : Run on CPU if True, else GPU
    conda_env       : Name of the conda environment with OCPCalculator

    Returns
    -------
    dict with the same keys as relax_mlff():
        label, miller, converged, energy_ev, fmax_final, atoms, slab,
        steps_taken, error
    """
    from ase.io import write as ase_write, read as ase_read

    atoms = slab_dict.get("atoms")
    if atoms is None:
        slab = slab_dict.get("slab")
        if slab is None:
            raise ValueError(f"slab_dict for {slab_dict['label']} has no atoms or slab.")
        atoms = AseAtomsAdaptor().get_atoms(slab)
    atoms = atoms.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.traj"
        out_path = Path(tmpdir) / "output.json"

        ase_write(str(in_path), atoms, format="traj")

        # Build the worker script inline
        worker_script = f"""
import sys, json, traceback
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

try:
    from fairchem.core import OCPCalculator
    atoms = read({str(in_path)!r}, format="traj")
    import numpy as _np
    atoms.set_tags(_np.ones(len(atoms), dtype=int))

    calc = OCPCalculator(checkpoint_path={checkpoint_path!r}, cpu={cpu!r})
    atoms.calc = calc

    step_log = []
    def _log():
        f = atoms.get_forces()
        step_log.append(float((f**2).sum(axis=1).max()**0.5))
    opt = BFGS(atoms, logfile=None)
    opt.attach(_log, interval=1)
    converged = opt.run(fmax={fmax!r}, steps={steps!r})

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float((forces**2).sum(axis=1).max()**0.5)

    write({str(out_path).replace("output.json", "relaxed.traj")!r}, atoms, format="traj")

    result = {{
        "converged": bool(converged),
        "energy_ev": energy,
        "fmax_final": fmax_final,
        "steps_taken": len(step_log),
        "error": None,
    }}
except Exception:
    result = {{
        "converged": False, "energy_ev": None, "fmax_final": None,
        "steps_taken": 0, "error": traceback.format_exc()
    }}

with open({str(out_path)!r}, "w") as fp:
    json.dump(result, fp)
"""
        script_path = Path(tmpdir) / "worker.py"
        script_path.write_text(worker_script)

        conda_bin = Path(sys.executable).parent.parent.parent / "envs" / conda_env / "bin" / "python"
        if not conda_bin.exists():
            # Try standard miniconda location
            conda_bin = Path.home() / "miniconda3" / "envs" / conda_env / "bin" / "python"
        if not conda_bin.exists():
            conda_bin = Path.home() / "anaconda3" / "envs" / conda_env / "bin" / "python"

        try:
            proc = subprocess.run(
                [str(conda_bin), str(script_path)],
                capture_output=True, text=True, timeout=3600,
            )
        except subprocess.TimeoutExpired:
            return {
                "label": slab_dict["label"], "miller": slab_dict["miller"],
                "converged": False, "energy_ev": None, "fmax_final": None,
                "atoms": atoms, "slab": None, "steps_taken": 0,
                "error": "Subprocess timed out after 1 hour",
            }

        if not out_path.exists():
            return {
                "label": slab_dict["label"], "miller": slab_dict["miller"],
                "converged": False, "energy_ev": None, "fmax_final": None,
                "atoms": atoms, "slab": None, "steps_taken": 0,
                "error": f"Worker failed:\nSTDOUT:{proc.stdout[-2000:]}\nSTDERR:{proc.stderr[-2000:]}",
            }

        result = json.loads(out_path.read_text())

        # Load relaxed structure if available
        relaxed_traj = Path(tmpdir) / "relaxed.traj"
        relaxed_atoms = None
        relaxed_slab = None
        if relaxed_traj.exists():
            from ase.io import read as ase_read2
            relaxed_atoms = ase_read2(str(relaxed_traj))
            try:
                relaxed_slab = AseAtomsAdaptor().get_structure(relaxed_atoms)
            except Exception:
                pass

        return {
            "label": slab_dict["label"],
            "miller": slab_dict["miller"],
            "converged": result["converged"],
            "energy_ev": result["energy_ev"],
            "fmax_final": result["fmax_final"],
            "atoms": relaxed_atoms or atoms,
            "slab": relaxed_slab,
            "steps_taken": result["steps_taken"],
            "error": result["error"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Bulk structure relaxation (positions only, fixed cell from MP)
# ─────────────────────────────────────────────────────────────────────────────

def relax_bulk_mlff(
    structure,
    model_name: str = "uma-s-1",
    fmax: float = 0.05,
    steps: int = 300,
    task_name: str = "omat",
    progress_callback=None,
) -> dict:
    """
    Relax a bulk structure (positions only, fixed cell) using a fairchem 2.x UMA model.

    Parameters
    ----------
    structure         : pymatgen Structure (lattice from MP, to be used as-is)
    model_name        : UMA model name
    fmax              : Force convergence (eV/Å)
    steps             : Max steps
    task_name         : "omat" is best for bulk; "oc20" also works
    progress_callback : Optional fn(step, fmax_now)

    Returns
    -------
    dict: formula, energy_ev, energy_per_atom_ev, n_atoms, converged,
          fmax_final, steps_taken, atoms, error
    """
    from ase.optimize import LBFGS

    atoms = AseAtomsAdaptor().get_atoms(structure)
    n_atoms = len(atoms)
    formula = structure.formula

    atoms.calc = _get_calculator(model_name, task_name)

    step_counter = [0]
    fmax_log = [None]

    def obs():
        step_counter[0] += 1
        forces = atoms.get_forces()
        fmax_now = float((forces ** 2).sum(axis=1).max() ** 0.5)
        fmax_log[0] = fmax_now
        if progress_callback:
            progress_callback(step_counter[0], fmax_now)

    opt = LBFGS(atoms, logfile=None)
    opt.attach(obs, interval=1)

    try:
        converged = opt.run(fmax=fmax, steps=steps)
    except Exception as e:
        return {
            "formula": formula, "energy_ev": None, "energy_per_atom_ev": None,
            "n_atoms": n_atoms, "converged": False, "fmax_final": fmax_log[0],
            "steps_taken": step_counter[0], "atoms": atoms, "error": str(e),
        }

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float((forces ** 2).sum(axis=1).max() ** 0.5)

    return {
        "formula": formula,
        "energy_ev": energy,
        "energy_per_atom_ev": energy / n_atoms,
        "n_atoms": n_atoms,
        "converged": bool(converged),
        "fmax_final": fmax_final,
        "steps_taken": step_counter[0],
        "atoms": atoms,
        "error": None,
    }


def relax_bulk_ocp_checkpoint(
    structure,
    checkpoint_path: str,
    fmax: float = 0.05,
    steps: int = 300,
    cpu: bool = True,
    conda_env: str = OCP_CONDA_ENV,
) -> dict:
    """
    Relax a bulk structure (positions only, fixed cell) using a local OCP checkpoint.
    Runs as subprocess under conda_env (fairchem_old_OCP).

    Returns
    -------
    dict: formula, energy_ev, energy_per_atom_ev, n_atoms, converged,
          fmax_final, steps_taken, atoms, error
    """
    from ase.io import write as ase_write

    atoms = AseAtomsAdaptor().get_atoms(structure)
    n_atoms = len(atoms)
    formula = structure.formula

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "bulk_input.traj"
        out_path = Path(tmpdir) / "bulk_result.json"

        ase_write(str(in_path), atoms, format="traj")

        worker_script = f"""
import sys, json, traceback
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

try:
    from fairchem.core import OCPCalculator
    atoms = read({str(in_path)!r}, format="traj")
    # GemNet-OC qint_tags=[1,2]: must have tag>=1 or quad-interaction block
    # produces 0 edges and crashes. Use tag=1 for all atoms.
    atoms.set_tags(np.ones(len(atoms), dtype=int))

    calc = OCPCalculator(checkpoint_path={checkpoint_path!r}, cpu={cpu!r})
    atoms.calc = calc

    step_log = []
    def _log():
        f = atoms.get_forces()
        step_log.append(float((f**2).sum(axis=1).max()**0.5))
    opt = BFGS(atoms, logfile=None)
    opt.attach(_log, interval=1)
    converged = opt.run(fmax={fmax!r}, steps={steps!r})

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float((forces**2).sum(axis=1).max()**0.5)

    write({str(out_path).replace("bulk_result.json", "bulk_relaxed.traj")!r}, atoms, format="traj")

    result = {{
        "converged": bool(converged),
        "energy_ev": energy,
        "energy_per_atom_ev": energy / len(atoms),
        "fmax_final": fmax_final,
        "steps_taken": len(step_log),
        "error": None,
    }}
except Exception:
    result = {{
        "converged": False, "energy_ev": None, "energy_per_atom_ev": None,
        "fmax_final": None, "steps_taken": 0, "error": traceback.format_exc()
    }}

with open({str(out_path)!r}, "w") as fp:
    json.dump(result, fp)
"""
        script_path = Path(tmpdir) / "bulk_worker.py"
        script_path.write_text(worker_script)

        conda_bin = Path(sys.executable).parent.parent.parent / "envs" / conda_env / "bin" / "python"
        if not conda_bin.exists():
            conda_bin = Path.home() / "miniconda3" / "envs" / conda_env / "bin" / "python"
        if not conda_bin.exists():
            conda_bin = Path.home() / "anaconda3" / "envs" / conda_env / "bin" / "python"

        try:
            proc = subprocess.run(
                [str(conda_bin), str(script_path)],
                capture_output=True, text=True, timeout=3600,
            )
        except subprocess.TimeoutExpired:
            return {
                "formula": formula, "energy_ev": None, "energy_per_atom_ev": None,
                "n_atoms": n_atoms, "converged": False, "fmax_final": None,
                "steps_taken": 0, "atoms": atoms, "error": "Subprocess timed out",
            }

        if not out_path.exists():
            return {
                "formula": formula, "energy_ev": None, "energy_per_atom_ev": None,
                "n_atoms": n_atoms, "converged": False, "fmax_final": None,
                "steps_taken": 0, "atoms": atoms,
                "error": f"Worker failed:\nSTDOUT:{proc.stdout[-1000:]}\nSTDERR:{proc.stderr[-1000:]}",
            }

        result = json.loads(out_path.read_text())

        relaxed_traj = Path(tmpdir) / "bulk_relaxed.traj"
        relaxed_atoms = None
        if relaxed_traj.exists():
            from ase.io import read as ase_read2
            relaxed_atoms = ase_read2(str(relaxed_traj))

        return {
            "formula": formula,
            "energy_ev": result["energy_ev"],
            "energy_per_atom_ev": result["energy_per_atom_ev"],
            "n_atoms": n_atoms,
            "converged": result["converged"],
            "fmax_final": result["fmax_final"],
            "steps_taken": result["steps_taken"],
            "atoms": relaxed_atoms or atoms,
            "error": result["error"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# DFT VASP input file generation
# ─────────────────────────────────────────────────────────────────────────────

# Default INCAR settings for slab relaxation
_DEFAULT_INCAR = {
    "SYSTEM": "slab relaxation",
    "ISTART": 0,
    "ICHARG": 2,
    "PREC": "Accurate",
    "ENCUT": 400,
    "EDIFF": 1e-5,
    "EDIFFG": -0.02,
    "NSW": 100,
    "IBRION": 2,
    "ISIF": 2,          # Relax ions only, not cell
    "POTIM": 0.3,
    "ISMEAR": 1,        # Methfessel-Paxton for metals
    "SIGMA": 0.1,
    "ALGO": "Fast",
    "GGA": "PE",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
    "NCORE": 4,
    "KPAR": 1,
}

_DEFAULT_KPOINTS = """\
Automatic mesh
0
Gamma
3 3 1
0 0 0
"""


def generate_vasp_inputs(
    slab_dicts: list[dict],
    incar_overrides: dict | None = None,
    kpoints_text: str | None = None,
    potcar_functional: str = "PBE",
) -> bytes:
    """
    Generate VASP input files for a list of slabs.

    Creates one subdirectory per slab with:
        POSCAR   — slab structure in VASP format
        INCAR    — DFT settings (slab-appropriate defaults)
        KPOINTS  — k-point mesh (Gamma-centered 3x3x1)

    Parameters
    ----------
    slab_dicts        : List of slab entries from surface_generator.generate_slabs()
    incar_overrides   : Dict to merge into/override the default INCAR settings
    kpoints_text      : Override KPOINTS file content (default: Gamma 3x3x1)
    potcar_functional : Functional tag for POTCAR note in INCAR (PBE or LDA)

    Returns
    -------
    bytes : ZIP archive of the directory tree, ready for st.download_button
    """
    from ase.io import write as ase_write

    incar = dict(_DEFAULT_INCAR)
    if incar_overrides:
        incar.update(incar_overrides)

    kpoints = kpoints_text or _DEFAULT_KPOINTS

    with tempfile.TemporaryDirectory() as tmpdir:
        for slab_dict in slab_dicts:
            if slab_dict.get("atoms") is None and slab_dict.get("slab") is None:
                continue

            # Sanitize label for directory name
            label = slab_dict["label"]
            safe_label = (
                label.replace(" ", "_")
                     .replace("(", "").replace(")", "")
                     .replace("/", "-")
                     .replace(".", "p")
            )
            slab_dir = Path(tmpdir) / safe_label
            slab_dir.mkdir(parents=True, exist_ok=True)

            # POSCAR
            atoms = slab_dict.get("atoms")
            if atoms is None:
                atoms = AseAtomsAdaptor().get_atoms(slab_dict["slab"])
            poscar_path = slab_dir / "POSCAR"
            ase_write(str(poscar_path), atoms, format="vasp")

            # INCAR
            incar_path = slab_dir / "INCAR"
            _write_incar(incar, str(incar_path))

            # KPOINTS
            kpoints_path = slab_dir / "KPOINTS"
            kpoints_path.write_text(kpoints)

            # POTCAR note (actual POTCAR requires local PP files)
            note_path = slab_dir / "POTCAR.note"
            elements = list(dict.fromkeys(atoms.get_chemical_symbols()))
            note_path.write_text(
                f"# Generate POTCAR by concatenating {potcar_functional} POTCARs for:\n"
                f"# {' '.join(elements)}\n"
                f"# Example: cat {' '.join(f'$VASP_PP_PATH/{potcar_functional}/{e}/POTCAR' for e in elements)} > POTCAR\n"
            )

        # Zip the directory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in Path(tmpdir).rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmpdir)
                    zf.write(file_path, arcname)
        zip_buffer.seek(0)
        return zip_buffer.read()


def _write_incar(incar_dict: dict, path: str):
    """Write INCAR file from dict."""
    lines = []
    for key, val in incar_dict.items():
        if isinstance(val, float):
            # Scientific notation for small floats
            if abs(val) < 0.01 and val != 0:
                lines.append(f"  {key} = {val:.0e}")
            else:
                lines.append(f"  {key} = {val}")
        else:
            lines.append(f"  {key} = {val}")
    Path(path).write_text("\n".join(lines) + "\n")


def get_default_incar() -> dict:
    """Return a copy of the default INCAR settings for display in UI."""
    return dict(_DEFAULT_INCAR)


_DEFAULT_INCAR_BULK = {
    "SYSTEM": "bulk relaxation",
    "ISTART": 0,
    "ICHARG": 2,
    "PREC": "Accurate",
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "EDIFFG": -0.02,
    "NSW": 200,
    "IBRION": 2,
    "ISIF": 2,          # positions only, fixed cell (lattice from MP)
    "POTIM": 0.3,
    "ISMEAR": 1,
    "SIGMA": 0.2,
    "ALGO": "Fast",
    "GGA": "PE",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
    "NCORE": 4,
    "KPAR": 1,
}

_DEFAULT_KPOINTS_BULK = """\
Automatic mesh
0
Gamma
7 7 7
0 0 0
"""


def generate_vasp_bulk_inputs(
    structures: list[tuple[str, object]],
    incar_overrides: dict | None = None,
    kpoints_text: str | None = None,
    potcar_functional: str = "PBE",
) -> bytes:
    """
    Generate VASP input files for bulk structure relaxation (positions only, ISIF=2).

    Parameters
    ----------
    structures  : list of (label, pymatgen Structure or ASE Atoms)
    incar_overrides : overrides merged into _DEFAULT_INCAR_BULK
    kpoints_text    : override KPOINTS content (default: Gamma 7×7×7)
    potcar_functional : PBE / LDA / PBEsol

    Returns
    -------
    bytes : ZIP archive
    """
    from ase.io import write as ase_write

    incar = dict(_DEFAULT_INCAR_BULK)
    if incar_overrides:
        incar.update(incar_overrides)

    kpoints = kpoints_text or _DEFAULT_KPOINTS_BULK

    with tempfile.TemporaryDirectory() as tmpdir:
        for label, structure in structures:
            safe_label = (
                label.replace(" ", "_").replace("/", "-").replace(".", "p")
                     .replace("(", "").replace(")", "")
            )
            struct_dir = Path(tmpdir) / safe_label
            struct_dir.mkdir(parents=True, exist_ok=True)

            # Get ASE atoms
            if hasattr(structure, "get_positions"):
                atoms = structure
            else:
                atoms = AseAtomsAdaptor().get_atoms(structure)

            poscar_path = struct_dir / "POSCAR"
            ase_write(str(poscar_path), atoms, format="vasp")

            incar_path = struct_dir / "INCAR"
            _write_incar(incar, str(incar_path))

            kpoints_path = struct_dir / "KPOINTS"
            kpoints_path.write_text(kpoints)

            elements = list(dict.fromkeys(atoms.get_chemical_symbols()))
            note_path = struct_dir / "POTCAR.note"
            note_path.write_text(
                f"# Generate POTCAR by concatenating {potcar_functional} POTCARs for:\n"
                f"# {' '.join(elements)}\n"
                f"# Example: cat {' '.join(f'$VASP_PP_PATH/{potcar_functional}/{e}/POTCAR' for e in elements)} > POTCAR\n"
            )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in Path(tmpdir).rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmpdir)
                    zf.write(file_path, arcname)
        zip_buffer.seek(0)
        return zip_buffer.read()
