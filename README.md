# SurfaceEnergy — Streamlit App for Surface Energy & Wulff Construction

A browser-based workflow for generating surface slabs, relaxing them with machine-learning force fields (MLFF), calculating surface energies via three thermodynamic methods, and building Wulff equilibrium crystal shapes.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Launching the App](#launching-the-app)
4. [Workflow — Tab by Tab](#workflow--tab-by-tab)
   - [Tab ① Load Structure](#tab--load-structure)
   - [Tab ② Surfaces](#tab--surfaces)
   - [Tab ③ Preview](#tab--preview)
   - [Tab ④ Optimize](#tab--optimize)
   - [Tab ⑤ References](#tab--references)
   - [Tab ⑥ Energies](#tab--energies)
   - [Tab ⑦ Wulff](#tab--wulff)
5. [Surface Energy Methods](#surface-energy-methods)
   - [Method 1 — Explicit Thermodynamic Decomposition](#method-1--explicit-thermodynamic-decomposition)
   - [Method 2 — Formation Energy Reference](#method-2--formation-energy-reference)
   - [Method 3 — Direct Linear Fit (N-limit)](#method-3--direct-linear-fit-n-limit)
6. [Supported MLFF Backends](#supported-mlff-backends)
7. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

```
Load bulk structure
       │
       ▼
Generate surface slabs (pymatgen SlabGenerator)
       │
       ▼
Preview 3-D structures (Plotly, in-browser)
       │
       ▼
Relax slabs with MLFF (fairchem UMA or local OCP checkpoint)
       │
       ▼
Calculate surface energies (Method 1 / 2 / 3)
       │
       ▼
Build Wulff construction → equilibrium crystal shape
```

All steps run in a single Streamlit web app — no coding required after launch.

---

## Installation

### Conda environment (recommended)

```bash
conda create -n surface_ui python=3.12
conda activate surface_ui

# Core scientific stack
pip install streamlit plotly pymatgen mp-api ase pandas numpy

# fairchem 2.x (UMA models)
pip install fairchem-core
```

> **Note:** If you also use a local OCP checkpoint (GemNet-OC etc.) that requires fairchem 1.x / `OCPCalculator`, create a **separate** conda environment (e.g. `fairchem_old_OCP`) and enter its path in Tab ④. The app spawns a subprocess automatically.

### Clone

```bash
git clone https://github.com/jinlijin6090/SurfaceEnergy.git
cd SurfaceEnergy
```

---

## Launching the App

```bash
conda activate surface_ui   # or whichever env has fairchem 2.x
streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502` in your browser.

**Remote server / SSH tunnel:**

```powershell
# Windows PowerShell (not MobaXterm)
ssh -L 8502:localhost:8502 username@server_ip
```

Then open `http://localhost:8502` on your local machine.

---

## Workflow — Tab by Tab

### Tab ① Load Structure

Load the **bulk unit cell** you want to study. Three input modes:

| Mode | How to use |
|------|-----------|
| **Upload file** | CIF, POSCAR/CONTCAR, or any pymatgen-readable file |
| **Materials Project ID** | Enter `mp-XXXXX`; requires a free MP API key |
| **Paste CIF text** | Paste raw CIF content into the text area |

After loading you will see:
- Chemical formula, space group, lattice parameters
- An interactive 3-D preview of the unit cell

> The loaded structure is used as the bulk reference for all subsequent steps.

---

### Tab ② Surfaces

Generate surface slab models from the bulk structure.

**Miller index selection:**

- *Auto-enumerate* — finds all symmetrically distinct (h k l) up to a chosen max index
- *Manual* — type specific Miller indices, e.g. `1,1,1  1,1,0  1,0,0`

**Slab parameters:**

| Parameter | Meaning | Typical value |
|-----------|---------|--------------|
| Min slab thickness | Minimum distance between periodic images along surface normal (Å) | 15 Å |
| Min vacuum thickness | Vacuum gap (Å) | 15 Å |
| Relaxation cutoff z | Å from top/bottom that are **free** to move; inner atoms fixed | 4 Å |
| Primitive cell | Reduce to primitive slab cell | ✓ |
| Center slab | Center slab in the vacuum | ✓ |
| Symmetrize | Try symmetric terminations first (both surfaces identical) | ✓ |

Click **Generate Slabs**. A table lists every termination (shift + index). Failed Miller indices are flagged with an error message.

---

### Tab ③ Preview

Select any slab from the dropdown to view it interactively:

- Atoms colored by CPK scheme
- Fixed atoms shown smaller (grey outline)
- Hover for element, position, and fix/free status

A bulk structure preview is also available for comparison.

---

### Tab ④ Optimize

Relax selected slabs with an MLFF. Select one or more slabs from the checkbox table, then choose a backend:

#### MLFF (fairchem UMA)

Select a **UMA model** from the dropdown:

| Model | Notes |
|-------|-------|
| `uma-s-1` | Small, fast, good for most metals |
| `uma-s-1p1` | Updated small model |
| `uma-m-1p1` | Medium, better accuracy |
| `esen-sm-*` | Equivariant models, various datasets |

Select **Task** (`oc20`, `omat`, `omol`, `odac`) matching your system.

> The model is loaded once and cached — subsequent slabs do **not** reload weights, so batch relaxation is fast.

#### MLFF (local OCP checkpoint)

- Enter the path to a `.pt` checkpoint file
- Specify the conda environment that contains `OCPCalculator` (default: `fairchem_old_OCP`)
- The app runs a temporary Python subprocess; results are returned as JSON

#### DFT (VASP file generation)

Generates `INCAR`, `POSCAR`, `KPOINTS` files (ISIF=2, γ-point centered 7×7×7 k-mesh for bulk; surface mesh for slabs) ready to submit to VASP.

After relaxation, results appear in a summary table:
- Converged (✓/✗)
- Final energy (eV)
- Final fmax (eV/Å)
- Steps taken

---

### Tab ⑤ References

Compute elemental reference energies and bulk formation energies needed by Methods 1 and 2.

**Section A — Elemental references:**

1. Elements are auto-detected from the loaded bulk
2. Fetch stable elemental structures from the Materials Project (requires MP API key)
3. Relax each elemental bulk with the same MLFF backend as your slabs
4. Energies appear as eV/atom — used as μᵢ° in Methods 1 and 2

**Section B — Bulk compound energy:**

Relax the bulk compound itself and enter the total energy per formula unit, or use the MLFF directly. The per-atom formation energy Eform is computed as:

```
E_form = E_compound/atom − Σ xᵢ μᵢ°
```

> **Important:** Always use the **same calculator** (same model, same task) for elemental references, bulk compound, and slab relaxations. Mixing DFT and MLFF energies gives unphysical results.

---

### Tab ⑥ Energies

Calculate surface energies from the relaxed slab data. Three methods are available (see [Surface Energy Methods](#surface-energy-methods) below).

**Common inputs (top of tab):**

- **Elemental references** — paste manually as `Pd=-5.18, Zn=-1.26` or auto-fill from Tab ⑤
- **Bulk compound energy per f.u.** — total energy of one formula unit of the compound
- **Bulk stoichiometry** — auto-detected from the loaded structure (e.g. `Pd=1, Zn=1`)
- **Formation energy per atom** — computed in Tab ⑤ or entered manually

Each method has its formula displayed next to the controls. Results from all three methods are stored **independently** and can be compared in Tab ⑦.

#### Method 3 — Thickness Series Generator

Inside the Method 3 expander, a "Generate thickness series" panel lets you:

1. Select one, several, or **all** Miller indices (multiselect with "— All —" option)
2. Set min/max thickness and step size
3. Choose a relaxation backend
4. Click **Generate & Relax thickness series**

The app generates slabs at each thickness, relaxes them, and fills the data table automatically. Each row = one thickness point. Method 3 then fits the series to extract γ.

---

### Tab ⑦ Wulff

Build the equilibrium crystal shape (Wulff construction) from computed surface energies.

**Energy source selector** — only shows methods that have results:
- Method 1, Method 2, Method 3, or Manual input

For each method, if multiple terminations (shifts) exist for the same Miller index, only the **lowest-energy termination** is used (physically correct: the equilibrium surface adopts the most stable termination).

Facets with NaN, infinite, non-positive energy, or invalid `(0 0 0)` Miller index are automatically dropped with a warning.

**Controls:**

| Control | Description |
|---------|-------------|
| Display energy unit | J/m² or eV/Å² |
| Bar chart sort | Energy ascending / facet name / area fraction |
| Custom facet colors | Color picker per facet |

**Output:**

- Interactive 3-D Wulff shape (Plotly Mesh3d, color per facet)
- Bar chart: surface energy per facet, annotated with area fraction %
- Metrics: number of exposed facets, weighted-average γ, anisotropy index, total surface area
- Detailed parameter table: γ (J/m²), γ (eV/Å²), area fraction %, exposed Y/N

---

## Surface Energy Methods

All methods assume **symmetric slab models** (both surfaces are identical), so the factor of 2 in the denominator cancels the contribution of two surfaces.

### Method 1 — Explicit Thermodynamic Decomposition

**Best for:** Stoichiometric slab terminations (slab has the same composition ratio as bulk).

$$\gamma = \frac{1}{2A}\left[E_{\rm slab} - \frac{a_n}{x_n}E_{\rm comp} - \sum_{j \neq n}\left(a_j - \frac{a_n}{x_n}x_j\right)\mu_j^\circ\right]$$

| Symbol | Meaning |
|--------|---------|
| A | Surface area of one face (Å²) |
| aᵢ | Number of element i atoms in the slab |
| xᵢ | Stoichiometry coefficient of element i in bulk compound |
| n | Normalising element (conventionally minority element) |
| E_comp | Bulk compound energy per formula unit (eV/f.u.) |
| μⱼ° | Elemental reference energy of element j (eV/atom) |

**Required inputs:** elemental references μᵢ°, bulk compound energy, bulk stoichiometry, normalising element.

---

### Method 2 — Formation Energy Reference

**Best for:** Off-stoichiometric terminations (slab composition deviates from bulk).

$$E_{\rm ref}(N) = \sum_i a_i\,\mu_i^\circ + N\cdot E_{\rm form}$$

$$\gamma = \frac{E_{\rm slab} - E_{\rm ref}}{2A}$$

| Symbol | Meaning |
|--------|---------|
| N | Total atoms in slab |
| E_form | Per-atom formation energy of bulk compound (eV/atom) |
| μᵢ° | Elemental reference energy (eV/atom) |

**Required inputs:** elemental references, per-atom formation energy.

---

### Method 3 — Direct Linear Fit (N-limit)

**Best for:** Systems where absolute reference energies are unavailable or unreliable (e.g. OC20-referenced MLFF checkpoints). No external references needed.

**Direct fit mode (recommended):**

$$E_{\rm slab}(N) = \varepsilon_{\rm bulk}\cdot N + 2\gamma A$$

Fit E_slab vs N atoms across a series of slab thicknesses:
- Slope → ε_bulk (bulk energy per atom, self-determined from the fit)
- Intercept → 2γA → γ = intercept / (2A)

**N-limit mode (with references):**

$$\delta E(N) = E_{\rm slab}(N) - E_{\rm ref}(N)$$

Fit δE vs N; γ = intercept / (2A). Requires elemental references and E_form.

**Minimum R² threshold** can be set; results below the threshold are flagged as not converged.

---

## Supported MLFF Backends

| Backend | Environment | Notes |
|---------|-------------|-------|
| fairchem UMA (uma-s-1, uma-m-1p1, …) | `fairchem_UMA_NEW` | fairchem ≥ 2.x; recommended |
| Local OCP checkpoint (.pt file) | `fairchem_old_OCP` | fairchem 1.x; GemNet-OC, SCN, etc. via subprocess |
| DFT (VASP) | any | File generation only; submit externally |

**OC20 checkpoint note:** GemNet-OC and similar OC20-trained models output *referenced* energies (not absolute DFT total energies). This makes Methods 1 and 2 give unphysical (often negative) surface energies. Use **Method 3 direct fit** in this case — references cancel in the linear fit.

---

## Tips & Troubleshooting

**Wulff construction fails with NaN**
- Caused by non-positive or NaN surface energies. Check that elemental references are negative (eV/atom from a stable solid), and that the bulk formation energy is also negative for a stable compound.
- Facets with `(0 0 0)` Miller index (invalid) are auto-dropped.

**Method 1 reports "Missing elemental reference for X"**
- The bulk stoichiometry field lists an element not in your elemental references. Check that stoichiometry was auto-detected correctly for the loaded bulk (it resets when you load a new structure).

**MLFF relaxation is slow on first slab, fast on subsequent**
- Normal: model weights are loaded and GPU-compiled on the first call, then cached. Subsequent slabs reuse the cached calculator with no reloading overhead.

**SSH port forwarding (Windows PowerShell)**
```powershell
# Run in PowerShell, not MobaXterm
ssh -L 8502:localhost:8502 username@server_ip
# Then open http://localhost:8502 in browser
```

**OCP subprocess crashes with "cannot reshape tensor of 0 elements"**
- Caused by a GemNet-OC requirement: all atoms must have `tag=1` for the quad-interaction block. The app sets this automatically.

**numpy version conflict in `fairchem_old_OCP`**
- numba requires numpy ≤ 2.3. Fix: `conda install "numpy<2.3"` in that environment.

---

## Citation

If you use this tool in your research, please cite:

> Jin Li, Gunnar Sly, Michael J. Janik. "Leveraging Pretrained Machine Learning Models for Surface Energy Prediction and Wulff Construction of Intermetallic Nanoparticles." *Penn State University.*

---

## License

MIT
