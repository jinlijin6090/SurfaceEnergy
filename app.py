"""
Surface Science Workflow UI
===========================
Streamlit application for:
  1. Load     — MPID or POSCAR → bulk structure
  2. Surfaces — Generate slabs (Miller index selection, thickness, vacuum)
  3. Preview  — 3D structure viewer for generated slabs
  4. Optimize — MLFF relaxation or DFT VASP file generation
  5. Energies — Surface energy calculation (3 methods)
  6. Wulff    — 3D Wulff construction + bar chart

Run:
    conda activate fairchem_UMA_NEW
    streamlit run app.py
"""

from __future__ import annotations

import io
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surface Science UI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────
_STATE_DEFAULTS = {
    "bulk": None,           # pymatgen Structure
    "bulk_label": "",       # chemical formula string
    "slabs": [],            # list of dicts from surface_generator.generate_slabs()
    "relaxed_slabs": [],    # list of dicts from optimizer.relax_mlff()
    "surface_energies": [], # list of SurfaceEnergyResult (legacy, kept for compat)
    "surface_energies_m1": [],  # results from Method 1
    "surface_energies_m2": [],  # results from Method 2
    "surface_energies_m3": [],  # results from Method 3
    "manual_energies": [],  # [(hkl, energy_j_m2)] from manual input in Tab 6
    "ref_elemental_structures": {},   # {element: {"structure", "mpid", "formula", "dft_energy_per_atom"}}
    "ref_elemental_energies": {},     # {element: eV/atom} — computed by MLFF/DFT
    "ref_compound_energy_per_atom": None,  # eV/atom
    "ref_formation_energy_per_atom": None, # eV/atom
    "ref_backend": "MLFF (fairchem UMA)",  # last used backend in Tab ⑤
    "ref_model": "uma-s-1",                # last used model
    "ref_checkpoint": "",                  # last used OCP checkpoint path
    "m3_thickness_series": None,           # generated thickness series for Method 3
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (avoid loading heavy deps if not needed)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _import_core():
    from core import structure_loader, surface_generator, optimizer, surface_energy, wulff
    from core import reference_calculator
    return structure_loader, surface_generator, optimizer, surface_energy, wulff, reference_calculator


# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("Surface Science Workflow")
tabs = st.tabs([
    "① Load",
    "② Surfaces",
    "③ Preview",
    "④ Optimize",
    "⑤ References",
    "⑥ Energies",
    "⑦ Wulff",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: LOAD
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Load Bulk Structure")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Option A: Materials Project MPID")
        api_key = st.text_input(
            "MP API Key",
            type="password",
            help="Get your key at https://materialsproject.org/api",
        )
        mpid_input = st.text_input("MPID", placeholder="e.g. mp-10905")
        load_mpid_btn = st.button("Load from MP", key="load_mpid")

        if load_mpid_btn:
            if not mpid_input:
                st.error("Please enter an MPID.")
            elif not api_key:
                st.error("Please provide your MP API key.")
            else:
                with st.spinner(f"Fetching {mpid_input} from Materials Project..."):
                    try:
                        loader, *_ = _import_core()
                        structure, label = loader.from_mpid(mpid_input.strip(), api_key.strip())
                        st.session_state["bulk"] = structure
                        st.session_state["bulk_label"] = label
                        st.success(f"Loaded: {label} ({mpid_input})")
                    except Exception as e:
                        st.error(f"Failed to load {mpid_input}: {e}")

    with col_right:
        st.subheader("Option B: Upload POSCAR / CONTCAR")
        uploaded = st.file_uploader(
            "Upload POSCAR or CONTCAR",
            type=["vasp", "poscar", "contcar", "txt", ""],
            help="Any VASP POSCAR/CONTCAR format",
        )
        load_poscar_btn = st.button("Load from file", key="load_poscar")

        if load_poscar_btn:
            if uploaded is None:
                st.error("Please upload a POSCAR file first.")
            else:
                with st.spinner("Parsing structure..."):
                    try:
                        loader, *_ = _import_core()
                        structure, label = loader.from_poscar(uploaded.read())
                        st.session_state["bulk"] = structure
                        st.session_state["bulk_label"] = label
                        st.success(f"Loaded: {label}")
                    except Exception as e:
                        st.error(f"Failed to parse file: {e}")

    # Display loaded structure info
    if st.session_state["bulk"] is not None:
        st.divider()
        st.subheader(f"Loaded: {st.session_state['bulk_label']}")
        loader, sg_mod, *_ = _import_core()
        info = loader.structure_info(st.session_state["bulk"])
        info_col1, info_col2 = st.columns(2)
        items = list(info.items())
        for key, val in items[:5]:
            info_col1.metric(key, val)
        for key, val in items[5:]:
            info_col2.metric(key, val)

        # 3D bulk preview
        st.subheader("Bulk Structure Preview")
        traces = sg_mod.structure_to_plotly_traces(st.session_state["bulk"])
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(aspectmode="data"),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(title="Elements"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: SURFACE GENERATION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Surface Generation")

    if st.session_state["bulk"] is None:
        st.info("Load a bulk structure in Tab ① first.")
    else:
        _, sg_mod, *_ = _import_core()

        st.subheader("Miller Index Selection")
        miller_mode = st.radio(
            "Mode",
            ["Auto-enumerate (symmetrically distinct)", "Manual input"],
            horizontal=True,
        )

        if miller_mode.startswith("Auto"):
            max_miller = st.slider("Max Miller index", 1, 5, 3)
            with st.spinner("Enumerating distinct Miller indices..."):
                all_millers = sg_mod.enumerate_miller_indices(
                    st.session_state["bulk"], max_index=max_miller
                )
            miller_labels = [f"({h} {k} {l})" for h, k, l in all_millers]
            selected_labels = st.multiselect(
                f"Select facets to generate ({len(all_millers)} available)",
                options=miller_labels,
                default=miller_labels[:min(6, len(miller_labels))],
            )
            selected_millers = [
                all_millers[miller_labels.index(s)] for s in selected_labels
            ]
        else:
            miller_text = st.text_area(
                "Enter Miller indices (one per line or comma-separated):",
                placeholder="1,1,1\n1,1,0\n1,0,0",
                height=120,
            )
            selected_millers = sg_mod.parse_miller_string(miller_text)
            if selected_millers:
                st.write(f"Parsed: {[f'({h} {k} {l})' for h,k,l in selected_millers]}")

        st.divider()
        st.subheader("Slab Parameters")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            min_slab_size = st.slider("Min slab thickness (Å)", 5.0, 50.0, 15.0, 1.0)
            z_cutoff = st.slider("Fixed layer z-cutoff (Å)", 1.0, 10.0, 4.0, 0.5,
                                  help="Atoms within z_cutoff Å of top/bottom are free; rest are fixed")
        with p_col2:
            min_vacuum = st.slider("Min vacuum (Å)", 10.0, 50.0, 15.0, 1.0)
        with p_col3:
            primitive = st.checkbox("Use primitive cell", value=True)
            center_slab = st.checkbox("Center slab in cell", value=True)

        st.divider()
        generate_btn = st.button(
            "Generate Slabs",
            type="primary",
            disabled=len(selected_millers) == 0,
        )

        if generate_btn and selected_millers:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_cb(i, total, hkl):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Processing ({hkl[0]} {hkl[1]} {hkl[2]})... [{i+1}/{total}]")

            with st.spinner("Generating slabs..."):
                try:
                    slabs = sg_mod.generate_slabs(
                        structure=st.session_state["bulk"],
                        miller_indices=selected_millers,
                        min_slab_size=min_slab_size,
                        min_vacuum_size=min_vacuum,
                        z_cutoff=z_cutoff,
                        primitive=primitive,
                        center_slab=center_slab,
                        progress_callback=progress_cb,
                    )
                    st.session_state["slabs"] = slabs
                    progress_bar.progress(1.0)
                    status_text.text("Done!")
                    st.success(f"Generated {len(slabs)} slab terminations.")
                except Exception as e:
                    st.error(f"Slab generation failed: {e}")
                    st.code(traceback.format_exc())

        # Show generated slabs table
        if st.session_state["slabs"]:
            st.divider()
            st.subheader(f"Generated Slabs ({len(st.session_state['slabs'])} terminations)")
            rows = []
            for s in st.session_state["slabs"]:
                rows.append({
                    "Label": s["label"],
                    "Miller": str(s["miller"]),
                    "Symmetric": "✓" if s["symmetric"] else "✗",
                    "Atoms": len(s["atoms"]) if s["atoms"] is not None else "—",
                    "Status": "OK" if s["error"] is None else f"Error: {s['error']}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: PREVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Surface Preview")

    if not st.session_state["slabs"]:
        st.info("Generate slabs in Tab ② first.")
    else:
        _, sg_mod, *_ = _import_core()
        valid_slabs = [s for s in st.session_state["slabs"] if s["error"] is None]

        if not valid_slabs:
            st.warning("No valid slabs to preview.")
        else:
            slab_labels = [s["label"] for s in valid_slabs]
            selected_label = st.selectbox("Select slab to preview", options=slab_labels)
            slab_dict = next(s for s in valid_slabs if s["label"] == selected_label)

            col_info, col_view = st.columns([1, 2])

            with col_info:
                st.subheader("Slab Info")
                hkl = slab_dict["miller"]
                atoms = slab_dict.get("atoms")
                st.write(f"**Miller index:** ({hkl[0]} {hkl[1]} {hkl[2]})")
                st.write(f"**Symmetric:** {'Yes' if slab_dict['symmetric'] else 'No (asymmetric)'}")
                if atoms is not None:
                    from ase.constraints import FixAtoms
                    n_fixed = sum(
                        len(c.index) for c in atoms.constraints if isinstance(c, FixAtoms)
                    )
                    n_free = len(atoms) - n_fixed
                    st.write(f"**Total atoms:** {len(atoms)}")
                    st.write(f"**Free atoms:** {n_free}")
                    st.write(f"**Fixed atoms:** {n_fixed}")
                    pos = atoms.get_positions()
                    st.write(f"**Slab height:** {pos[:,2].max() - pos[:,2].min():.2f} Å")
                    st.write(f"**Cell z:** {atoms.cell[2,2]:.2f} Å")

                if slab_dict.get("slab") is not None:
                    slab = slab_dict["slab"]
                    st.write(f"**Surface area:** {slab.surface_area:.2f} Å²")

                st.divider()
                st.write("**Composition:**")
                if atoms is not None:
                    from collections import Counter
                    comp = Counter(atoms.get_chemical_symbols())
                    for elem, count in sorted(comp.items()):
                        st.write(f"  {elem}: {count}")

            with col_view:
                st.subheader("3D Structure")
                show_fixed = st.checkbox("Color fixed/free atoms differently", value=True)

                traces = sg_mod.slab_to_plotly_traces(slab_dict)
                if traces:
                    fig = go.Figure(data=traces)
                    fig.update_layout(
                        scene=dict(
                            aspectmode="data",
                            xaxis_title="x (Å)",
                            yaxis_title="y (Å)",
                            zaxis_title="z (Å)",
                        ),
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(title="Elements"),
                        title=f"Slab: {selected_label}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Download POSCAR
            st.divider()
            if atoms is not None:
                import tempfile
                from ase.io import write as ase_write
                buf = io.StringIO()
                with tempfile.NamedTemporaryFile(suffix=".vasp", mode="w", delete=False) as f:
                    tmp_path = f.name
                ase_write(tmp_path, atoms, format="vasp")
                with open(tmp_path, "r") as f:
                    poscar_content = f.read()
                st.download_button(
                    "Download POSCAR",
                    data=poscar_content,
                    file_name=f"POSCAR_{selected_label[:30].replace(' ', '_')}.vasp",
                    mime="text/plain",
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: OPTIMIZE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Structure Optimization")

    if not st.session_state["slabs"]:
        st.info("Generate slabs in Tab ② first.")
    else:
        _, _, opt_mod, *_ = _import_core()
        valid_slabs = [s for s in st.session_state["slabs"] if s["error"] is None]

        opt_mode = st.radio(
            "Optimization mode",
            ["MLFF (fairchem UMA)", "MLFF (local OCP checkpoint)", "DFT (generate VASP files)"],
            horizontal=True,
        )
        st.session_state["ref_backend"] = opt_mode

        # Slab selection
        st.subheader("Select slabs to optimize")
        all_labels = [s["label"] for s in valid_slabs]
        selected_labels = st.multiselect(
            "Slabs",
            options=all_labels,
            default=all_labels[:min(3, len(all_labels))],
        )
        selected_slabs = [s for s in valid_slabs if s["label"] in selected_labels]

        st.divider()

        # ── MLFF (UMA) mode ──
        if opt_mode == "MLFF (fairchem UMA)":
            st.subheader("MLFF Settings")
            mlff_col1, mlff_col2, mlff_col3 = st.columns(3)
            with mlff_col1:
                model_name = st.selectbox("UMA model", opt_mod.AVAILABLE_UMA_MODELS)
                task_name = st.selectbox("Task", ["oc20", "omol", "omat", "odac"])
                st.session_state["ref_model"] = model_name
            with mlff_col2:
                fmax = st.number_input("fmax (eV/Å)", value=0.05, min_value=0.001, max_value=1.0, step=0.01)
                steps = st.number_input("Max steps", value=200, min_value=10, max_value=2000, step=10)

            run_mlff_btn = st.button(
                "Run MLFF Relaxation",
                type="primary",
                disabled=len(selected_slabs) == 0,
            )

            if run_mlff_btn and selected_slabs:
                relaxed_results = []
                progress_bar = st.progress(0)
                result_container = st.empty()

                for i, slab_dict in enumerate(selected_slabs):
                    step_ph = st.empty()
                    step_ph.text(f"Relaxing: {slab_dict['label']}...")

                    step_log = [0]
                    def step_cb(step, fmax_now, ph=step_ph, lbl=slab_dict["label"]):
                        ph.text(f"{lbl} | step {step} | fmax={fmax_now:.4f} eV/Å")
                        step_log[0] = step

                    try:
                        result = opt_mod.relax_mlff(
                            slab_dict=slab_dict,
                            model_name=model_name,
                            fmax=float(fmax),
                            steps=int(steps),
                            task_name=task_name,
                            progress_callback=step_cb,
                        )
                        relaxed_results.append(result)
                    except Exception as e:
                        relaxed_results.append({
                            "label": slab_dict["label"],
                            "miller": slab_dict["miller"],
                            "converged": False,
                            "energy_ev": None,
                            "error": str(e),
                        })

                    progress_bar.progress((i + 1) / len(selected_slabs))

                st.session_state["relaxed_slabs"] = relaxed_results

                # Summary table
                st.subheader("Relaxation Results")
                rows = []
                for r in relaxed_results:
                    rows.append({
                        "Label": r["label"],
                        "Converged": "✓" if r.get("converged") else "✗",
                        "Energy (eV)": f"{r['energy_ev']:.4f}" if r.get("energy_ev") is not None else "—",
                        "fmax (eV/Å)": f"{r.get('fmax_final', 0):.4f}" if r.get("fmax_final") else "—",
                        "Steps": r.get("steps_taken", "—"),
                        "Error": r.get("error") or "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ── MLFF (local OCP checkpoint) mode ──
        elif opt_mode == "MLFF (local OCP checkpoint)":
            st.subheader("Local OCP Checkpoint Settings")
            st.info(
                "Uses `OCPCalculator` (fairchem 1.x / GemNet-OC / GNOc) via the "
                f"`{opt_mod.OCP_CONDA_ENV}` conda environment. "
                "Supports `.pt` checkpoint files trained with the OCP framework."
            )
            ocp_col1, ocp_col2 = st.columns(2)
            with ocp_col1:
                checkpoint_path = st.text_input(
                    "Checkpoint path (.pt)",
                    value="/mnt/E/PdAuSb_Surface_energy/PdAuSb_surface_energy/new/gnoc_oc22_oc20_all_s2ef.pt",
                    help="Absolute path to the OCP .pt checkpoint file",
                )
                use_cpu = st.checkbox("Run on CPU", value=True)
                st.session_state["ref_checkpoint"] = checkpoint_path
            with ocp_col2:
                fmax_ocp = st.number_input("fmax (eV/Å)", value=0.05, min_value=0.001, max_value=1.0, step=0.01, key="ocp_fmax")
                steps_ocp = st.number_input("Max steps", value=200, min_value=10, max_value=2000, step=10, key="ocp_steps")

            ckpt_exists = checkpoint_path and __import__("os").path.isfile(checkpoint_path)
            if checkpoint_path and not ckpt_exists:
                st.warning(f"Checkpoint not found: `{checkpoint_path}`")

            run_ocp_btn = st.button(
                "Run OCP Relaxation",
                type="primary",
                disabled=len(selected_slabs) == 0 or not ckpt_exists,
            )

            if run_ocp_btn and selected_slabs and ckpt_exists:
                relaxed_results = []
                progress_bar = st.progress(0)

                for i, slab_dict in enumerate(selected_slabs):
                    step_ph = st.empty()
                    step_ph.text(f"Relaxing (subprocess): {slab_dict['label']}...")
                    try:
                        result = opt_mod.relax_ocp_checkpoint(
                            slab_dict=slab_dict,
                            checkpoint_path=checkpoint_path,
                            fmax=float(fmax_ocp),
                            steps=int(steps_ocp),
                            cpu=bool(use_cpu),
                        )
                        relaxed_results.append(result)
                        if result["error"]:
                            step_ph.error(f"Error: {result['error'][:300]}")
                        else:
                            step_ph.text(
                                f"{slab_dict['label']} done — "
                                f"E={result['energy_ev']:.4f} eV, "
                                f"fmax={result['fmax_final']:.4f}, "
                                f"steps={result['steps_taken']}, "
                                f"converged={result['converged']}"
                            )
                    except Exception as e:
                        relaxed_results.append({
                            "label": slab_dict["label"],
                            "miller": slab_dict["miller"],
                            "converged": False,
                            "energy_ev": None,
                            "fmax_final": None,
                            "atoms": None,
                            "slab": None,
                            "steps_taken": 0,
                            "error": str(e),
                        })

                    progress_bar.progress((i + 1) / len(selected_slabs))

                st.session_state["relaxed_slabs"] = relaxed_results

                st.subheader("Relaxation Results")
                rows = []
                for r in relaxed_results:
                    rows.append({
                        "Label": r["label"],
                        "Converged": "✓" if r.get("converged") else "✗",
                        "Energy (eV)": f"{r['energy_ev']:.4f}" if r.get("energy_ev") is not None else "—",
                        "fmax (eV/Å)": f"{r.get('fmax_final', 0):.4f}" if r.get("fmax_final") else "—",
                        "Steps": r.get("steps_taken", "—"),
                        "Error": r.get("error") or "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ── DFT mode ──
        else:
            st.subheader("VASP Input Settings")
            dft_col1, dft_col2 = st.columns(2)

            with dft_col1:
                st.write("**INCAR overrides** (key = value, one per line):")
                incar_text = st.text_area(
                    "INCAR overrides",
                    value="ENCUT = 400\nEDIFFG = -0.02\nNSW = 100\nISMEAR = 1\nSIGMA = 0.1",
                    height=150,
                    label_visibility="collapsed",
                )
                potcar_functional = st.selectbox("POTCAR functional", ["PBE", "LDA", "PBEsol"])

            with dft_col2:
                st.write("**KPOINTS** (full file content):")
                kpoints_text = st.text_area(
                    "KPOINTS",
                    value="Automatic mesh\n0\nGamma\n3 3 1\n0 0 0\n",
                    height=150,
                    label_visibility="collapsed",
                )

            # Parse INCAR overrides
            incar_overrides = {}
            for line in incar_text.strip().split("\n"):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    # Try to convert to int/float
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    incar_overrides[key] = val

            gen_dft_btn = st.button(
                "Generate VASP Files (ZIP)",
                type="primary",
                disabled=len(selected_slabs) == 0,
            )

            if gen_dft_btn and selected_slabs:
                with st.spinner("Generating VASP input files..."):
                    try:
                        zip_bytes = opt_mod.generate_vasp_inputs(
                            slab_dicts=selected_slabs,
                            incar_overrides=incar_overrides if incar_overrides else None,
                            kpoints_text=kpoints_text,
                            potcar_functional=potcar_functional,
                        )
                        st.success(f"Generated inputs for {len(selected_slabs)} slabs.")
                        st.download_button(
                            "Download vasp_inputs.zip",
                            data=zip_bytes,
                            file_name="vasp_inputs.zip",
                            mime="application/zip",
                        )
                    except Exception as e:
                        st.error(f"Failed to generate VASP inputs: {e}")
                        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: REFERENCE CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Reference Energy Calculations")
    st.caption(
        "Compute elemental reference energies (μ°) and bulk compound formation energy "
        "for use in Surface Energy Methods 1 and 2."
    )

    if st.session_state["bulk"] is None:
        st.info("Load a bulk structure in Tab ① first.")
    else:
        _, _, ref_opt_mod, _, _, _ = _import_core()

        bulk_struct = st.session_state["bulk"]
        elements_in_bulk = sorted({str(el) for el in bulk_struct.composition.elements})

        st.info(f"Elements detected in loaded structure: **{', '.join(elements_in_bulk)}**")

        # ── Backend selection (synced with Tab ④) ──
        st.subheader("Calculation backend")
        default_backend_idx = ["MLFF (fairchem UMA)", "MLFF (local OCP checkpoint)", "DFT (generate VASP files)"].index(
            st.session_state.get("ref_backend", "MLFF (fairchem UMA)")
        ) if st.session_state.get("ref_backend") in ["MLFF (fairchem UMA)", "MLFF (local OCP checkpoint)", "DFT (generate VASP files)"] else 0

        ref_backend = st.radio(
            "Backend (should match Tab ④ choice)",
            ["MLFF (fairchem UMA)", "MLFF (local OCP checkpoint)", "DFT (generate VASP files)"],
            index=default_backend_idx,
            horizontal=True,
            key="ref_backend_radio",
        )

        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            if ref_backend == "MLFF (fairchem UMA)":
                ref_model = st.selectbox(
                    "UMA model", ref_opt_mod.AVAILABLE_UMA_MODELS,
                    index=ref_opt_mod.AVAILABLE_UMA_MODELS.index(st.session_state.get("ref_model", "uma-s-1"))
                    if st.session_state.get("ref_model") in ref_opt_mod.AVAILABLE_UMA_MODELS else 0,
                    key="ref_model_sel",
                )
                ref_task = st.selectbox("Task", ["omat", "oc20", "omol"], key="ref_task")
            elif ref_backend == "MLFF (local OCP checkpoint)":
                ref_checkpoint = st.text_input(
                    "Checkpoint path (.pt)",
                    value=st.session_state.get("ref_checkpoint", ""),
                    key="ref_ckpt_path",
                )
                ref_cpu = st.checkbox("Run on CPU", value=True, key="ref_cpu")
        with ref_col2:
            ref_fmax = st.number_input("fmax (eV/Å)", value=0.05, min_value=0.001, step=0.01, key="ref_fmax")
            ref_steps = st.number_input("Max steps", value=300, min_value=10, max_value=2000, step=10, key="ref_steps")

        st.divider()

        # ══ SECTION A: Elemental references ══
        st.subheader("A. Elemental reference structures (μ°)")

        api_key_ref = st.text_input(
            "MP API Key (for fetching elemental structures)",
            type="password",
            key="ref_api_key",
            help="Same key as Tab ①",
        )

        fetch_btn = st.button("Fetch elemental structures from MP", key="fetch_elem_btn",
                              disabled=not api_key_ref)
        if fetch_btn and api_key_ref:
            with st.spinner(f"Fetching {elements_in_bulk} from Materials Project..."):
                try:
                    from core.reference_calculator import fetch_elemental_structures
                    elem_structs = fetch_elemental_structures(elements_in_bulk, api_key_ref)
                    st.session_state["ref_elemental_structures"] = elem_structs
                    fetched_labels = ", ".join(f"{el} ({d['mpid']})" for el, d in elem_structs.items())
                    st.success(f"Fetched: {fetched_labels}")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

        # Show fetched structures
        elem_structs = st.session_state.get("ref_elemental_structures", {})
        if elem_structs:
            fetch_rows = []
            for el, d in elem_structs.items():
                mlff_e = st.session_state.get("ref_elemental_energies", {}).get(el)
                fetch_rows.append({
                    "Element": el,
                    "MPID": d["mpid"],
                    "Formula": d["formula"],
                    "DFT E/atom (MP)": f"{d['dft_energy_per_atom']:.4f}",
                    "MLFF E/atom": f"{mlff_e:.5f}" if mlff_e is not None else "—",
                })
            st.dataframe(pd.DataFrame(fetch_rows), use_container_width=True)

            if ref_backend == "DFT (generate VASP files)":
                if st.button("Generate VASP files for elemental bulks", key="gen_elem_vasp"):
                    with st.spinner("Generating..."):
                        try:
                            struct_list = [(el, d["structure"]) for el, d in elem_structs.items()]
                            zip_bytes = ref_opt_mod.generate_vasp_bulk_inputs(struct_list)
                            st.download_button(
                                "Download elemental_bulks.zip",
                                data=zip_bytes,
                                file_name="elemental_bulks.zip",
                                mime="application/zip",
                            )
                        except Exception as e:
                            st.error(f"Failed: {e}")
            else:
                relax_elem_btn = st.button(
                    "Relax all elemental structures",
                    type="primary",
                    key="relax_elem_btn",
                )
                if relax_elem_btn:
                    elemental_energies = {}
                    for el, d in elem_structs.items():
                        with st.spinner(f"Relaxing {el} ({d['formula']})..."):
                            try:
                                if ref_backend == "MLFF (fairchem UMA)":
                                    result = ref_opt_mod.relax_bulk_mlff(
                                        structure=d["structure"],
                                        model_name=ref_model,
                                        fmax=float(ref_fmax),
                                        steps=int(ref_steps),
                                        task_name=ref_task,
                                    )
                                else:
                                    result = ref_opt_mod.relax_bulk_ocp_checkpoint(
                                        structure=d["structure"],
                                        checkpoint_path=ref_checkpoint,
                                        fmax=float(ref_fmax),
                                        steps=int(ref_steps),
                                        cpu=ref_cpu,
                                    )
                                if result["error"]:
                                    st.warning(f"{el}: {result['error'][:200]}")
                                else:
                                    elemental_energies[el] = result["energy_per_atom_ev"]
                                    st.success(
                                        f"{el}: E = {result['energy_per_atom_ev']:.5f} eV/atom, "
                                        f"fmax={result['fmax_final']:.4f}, converged={result['converged']}"
                                    )
                            except Exception as e:
                                st.warning(f"{el}: {e}")

                    if elemental_energies:
                        st.session_state["ref_elemental_energies"] = elemental_energies

        # Manual elemental energy input
        st.write("**Manual elemental energy input** (overrides MLFF results above):")
        manual_elem_text = st.text_area(
            "element = eV/atom, one per line",
            value="\n".join(
                f"{el} = {e:.5f}"
                for el, e in st.session_state.get("ref_elemental_energies", {}).items()
            ),
            height=100,
            placeholder="Cu = -3.72000\nPd = -5.17900",
            key="manual_elem_text",
        )
        if st.button("Save elemental energies", key="save_elem_btn"):
            parsed = {}
            for line in manual_elem_text.strip().split("\n"):
                if "=" in line:
                    el, _, val = line.partition("=")
                    try:
                        parsed[el.strip()] = float(val.strip())
                    except ValueError:
                        pass
            if parsed:
                st.session_state["ref_elemental_energies"] = parsed
                st.success(f"Saved: {parsed}")

        st.divider()

        # ══ SECTION B: Compound bulk ══
        st.subheader("B. Bulk compound relaxation")
        st.write(f"Structure: **{st.session_state['bulk_label']}** ({len(bulk_struct)} atoms)")

        n_bulk_atoms = len(bulk_struct)
        st.caption(f"Cell from MP will be kept fixed. Only atomic positions are relaxed.")

        relax_compound_btn = st.button(
            "Relax compound bulk",
            type="primary",
            key="relax_compound_btn",
            disabled=(ref_backend == "DFT (generate VASP files)"),
        )

        if ref_backend == "DFT (generate VASP files)":
            if st.button("Generate VASP file for compound bulk", key="gen_compound_vasp"):
                with st.spinner("Generating..."):
                    try:
                        zip_bytes = ref_opt_mod.generate_vasp_bulk_inputs(
                            [(st.session_state["bulk_label"], bulk_struct)]
                        )
                        st.download_button(
                            "Download compound_bulk.zip",
                            data=zip_bytes,
                            file_name="compound_bulk.zip",
                            mime="application/zip",
                        )
                    except Exception as e:
                        st.error(f"Failed: {e}")

        if relax_compound_btn and ref_backend != "DFT (generate VASP files)":
            with st.spinner(f"Relaxing {st.session_state['bulk_label']}..."):
                try:
                    if ref_backend == "MLFF (fairchem UMA)":
                        result = ref_opt_mod.relax_bulk_mlff(
                            structure=bulk_struct,
                            model_name=ref_model,
                            fmax=float(ref_fmax),
                            steps=int(ref_steps),
                            task_name=ref_task,
                        )
                    else:
                        result = ref_opt_mod.relax_bulk_ocp_checkpoint(
                            structure=bulk_struct,
                            checkpoint_path=ref_checkpoint,
                            fmax=float(ref_fmax),
                            steps=int(ref_steps),
                            cpu=ref_cpu,
                        )
                    if result["error"]:
                        st.error(f"Relaxation failed: {result['error'][:400]}")
                    else:
                        e_per_atom = result["energy_per_atom_ev"]
                        st.session_state["ref_compound_energy_per_atom"] = e_per_atom
                        st.success(
                            f"E = {result['energy_ev']:.5f} eV total | "
                            f"{e_per_atom:.5f} eV/atom | "
                            f"fmax={result['fmax_final']:.4f} | "
                            f"converged={result['converged']}"
                        )
                except Exception as e:
                    st.error(f"{e}")
                    st.code(traceback.format_exc())

        compound_e_per_atom_input = st.number_input(
            "Compound energy per atom (eV/atom) — enter manually or from relaxation above",
            value=st.session_state.get("ref_compound_energy_per_atom") or -5.0,
            step=0.001,
            format="%.5f",
            key="compound_e_per_atom_input",
        )
        if st.button("Save compound energy", key="save_compound_btn"):
            st.session_state["ref_compound_energy_per_atom"] = compound_e_per_atom_input
            st.success(f"Saved: {compound_e_per_atom_input:.5f} eV/atom")

        st.divider()

        # ══ SECTION C: Formation energy ══
        st.subheader("C. Formation energy")

        elemental_energies_saved = st.session_state.get("ref_elemental_energies", {})
        compound_e_saved = st.session_state.get("ref_compound_energy_per_atom")

        if not elemental_energies_saved:
            st.warning("Elemental energies not yet computed (Section A).")
        elif compound_e_saved is None:
            st.warning("Compound bulk energy not yet computed (Section B).")
        else:
            from core.reference_calculator import calc_formation_energy_per_atom

            comp_dict = {str(el): float(bulk_struct.composition[el]) for el in bulk_struct.composition.elements}

            e_form = calc_formation_energy_per_atom(
                compound_energy_per_atom_ev=compound_e_saved,
                compound_composition=comp_dict,
                elemental_energies_ev=elemental_energies_saved,
            )
            st.session_state["ref_formation_energy_per_atom"] = e_form

            st.metric(
                "Formation energy per atom",
                f"{e_form:.5f} eV/atom",
                delta=None,
                help="Negative = stable compound",
            )
            st.caption(
                f"E_form = E_compound/atom − Σ x_i μ_i°  =  {compound_e_saved:.5f} − "
                + " − ".join(
                    f"{comp_dict.get(el,0)/sum(comp_dict.values()):.3f}×{elemental_energies_saved[el]:.5f}"
                    for el in elemental_energies_saved if el in comp_dict
                )
            )

            st.divider()
            st.subheader("Apply to Surface Energy Methods 1 & 2")
            st.write("These values will be pre-filled in Tab ⑥:")
            st.json({
                "elemental_refs (eV/atom)": {k: round(v, 5) for k, v in elemental_energies_saved.items()},
                "compound_energy_per_atom (eV)": round(compound_e_saved, 5),
                "formation_energy_per_atom (eV)": round(e_form, 5),
            })


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: SURFACE ENERGIES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Surface Energy Calculation")
    st.caption(
        "Three methods from: Li, Sly, Janik — "
        "*Leveraging Pretrained ML Models for Surface Energy Prediction and Wulff Construction*"
    )

    _, _, _, se_mod, *_ = _import_core()

    # ── Manual / direct energy input ──
    st.subheader("Quick: Manual Energy Entry")
    st.write("Paste known surface energies (J/m²) directly — bypasses all calculation methods:")

    if st.session_state["slabs"]:
        unique_millers_se = sorted({s["miller"] for s in st.session_state["slabs"] if s["error"] is None})
        default_rows = [{"h": h, "k": k, "l": l, "energy_j_m2": 1.0} for h, k, l in unique_millers_se]
    else:
        default_rows = [
            {"h": 1, "k": 0, "l": 0, "energy_j_m2": 1.0},
            {"h": 1, "k": 1, "l": 0, "energy_j_m2": 1.2},
            {"h": 1, "k": 1, "l": 1, "energy_j_m2": 0.9},
        ]

    edited_df = st.data_editor(
        pd.DataFrame(default_rows),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "h": st.column_config.NumberColumn("h", format="%d"),
            "k": st.column_config.NumberColumn("k", format="%d"),
            "l": st.column_config.NumberColumn("l", format="%d"),
            "energy_j_m2": st.column_config.NumberColumn("Surface energy (J/m²)", format="%.4f", min_value=0.0),
        },
    )
    if st.button("Save manual energies → Wulff", type="primary", key="save_manual"):
        manual = []
        for _, row in edited_df.iterrows():
            try:
                manual.append(((int(row["h"]), int(row["k"]), int(row["l"])), float(row["energy_j_m2"])))
            except Exception:
                continue
        st.session_state["manual_energies"] = manual
        st.success(f"Saved {len(manual)} surface energies.")

    st.divider()

    # ── Shared reference inputs (used by all three methods) ──
    with st.expander("Reference energies (required for Methods 1, 2, 3)", expanded=False):
        st.markdown("""
        Enter elemental reference energies μ_i° (eV/atom) — total energy per atom of each
        stable elemental solid calculated with the **same calculator** (DFT or MLFF) as the slabs.
        """)
        ref_text = st.text_area(
            "Elemental references (element = eV/atom, one per line)",
            placeholder="Pd = -5.179\nAl = -3.745\nCu = -3.720",
            height=120,
            key="ref_text",
        )
        elemental_refs = {}
        for line in ref_text.strip().split("\n"):
            if "=" in line:
                el, _, val = line.partition("=")
                try:
                    elemental_refs[el.strip()] = float(val.strip())
                except ValueError:
                    pass

        # Auto-fill from Tab ⑤ if available and text box is empty
        if not elemental_refs and st.session_state.get("ref_elemental_energies"):
            elemental_refs = st.session_state["ref_elemental_energies"]
            st.info("Elemental references auto-filled from Tab ⑤ References.")

        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            bulk_e_fu = st.number_input(
                "Bulk compound energy per formula unit (eV/f.u.)",
                value=-8.92,
                step=0.01,
                key="bulk_e_fu",
                help="E_comp: total DFT/MLFF energy of bulk compound divided by number of formula units",
            )
            st.caption(
                "Tip: use compound_energy_per_atom × N_atoms_per_f.u. from Tab ⑤ Section B."
            )
            # Auto-detect stoichiometry from loaded bulk structure; reset when bulk changes
            _bulk_stoich_default = ""
            _bulk_formula_now = ""
            if st.session_state.get("bulk") is not None:
                try:
                    _comp = st.session_state["bulk"].composition
                    _reduced = _comp.reduced_composition
                    _bulk_formula_now = _reduced.reduced_formula
                    _bulk_stoich_default = ", ".join(
                        f"{el}={int(amt)}" for el, amt in _reduced.items()
                    )
                except Exception:
                    pass
            # Reset session state when bulk formula changes so value= takes effect
            if st.session_state.get("_stoich_bulk_formula") != _bulk_formula_now:
                st.session_state["_stoich_bulk_formula"] = _bulk_formula_now
                st.session_state["stoich_text"] = _bulk_stoich_default
            stoich_text = st.text_input(
                "Bulk stoichiometry",
                value=_bulk_stoich_default,
                help="e.g. 'Pd=1, Zn=1' for PdZn; 'Pd=3, Zn=1' for Pd3Zn",
                key="stoich_text",
            )
        with ref_col2:
            e_form_per_atom = st.number_input(
                "Formation energy per atom E_form (eV/atom)",
                value=st.session_state.get("ref_formation_energy_per_atom") or -0.48,
                step=0.01,
                key="e_form",
                help="(E_compound/f.u. - Σ x_i×μ_i°) / N_atoms_per_f.u. — negative for stable compounds",
            )

        # Parse stoichiometry
        bulk_stoich = {}
        for tok in stoich_text.split(","):
            tok = tok.strip()
            if "=" in tok:
                el, _, x = tok.partition("=")
                try:
                    bulk_stoich[el.strip()] = float(x.strip())
                except ValueError:
                    pass

        # Display relaxed slab energies if available
        if st.session_state.get("relaxed_slabs"):
            st.write("**Relaxed slab energies from Tab ④:**")
            rows_relax = [
                {
                    "Label": r["label"],
                    "Miller": str(r["miller"]),
                    "E_slab (eV)": f"{r['energy_ev']:.5f}" if r.get("energy_ev") else "—",
                    "Converged": "✓" if r.get("converged") else "✗",
                }
                for r in st.session_state["relaxed_slabs"] if r.get("energy_ev") is not None
            ]
            if rows_relax:
                st.dataframe(pd.DataFrame(rows_relax), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # METHOD 1 — Explicit thermodynamic decomposition
    # ─────────────────────────────────────────────────────────────────
    with st.expander("Method 1 — Explicit thermodynamic decomposition"):
        m1_left, m1_right = st.columns([1, 1])
        with m1_left:
            st.markdown("**Formula:**")
            st.latex(r"""\gamma = \frac{1}{2A}\!\left[E_{\rm slab}
  - \frac{a_n}{x_n} E_{\rm comp}
  - \sum_{j \neq n}\!\left(a_j - \frac{a_n}{x_n} x_j\right)\mu_j^\circ\right]""")
            st.markdown(r"""
**Notation:**
- $a_i$ = number of element $i$ atoms in slab
- $x_i$ = stoichiometry in bulk compound
- $n$ = normalising element; $\lambda = a_n/x_n$ formula units
- $E_{\rm comp}$ = bulk compound energy (eV/f.u.)
- $\mu_j^\circ$ = elemental reference (eV/atom)
- $A$ = surface area (Å²)

**When to use:** Stoichiometric slab terminations (bulk-like composition).
            """)

        with m1_right:
            m1_norm_elem = st.text_input(
                "Normalising element (n)",
                value=list(bulk_stoich.keys())[-1] if bulk_stoich else "Al",
                key="m1_norm",
                help="Element used to count formula units. Conventionally the minority element.",
            )

        st.write("**Slab energies to calculate (one row per slab):**")
        m1_default = []
        for r in st.session_state.get("relaxed_slabs", []):
            if r.get("energy_ev") is not None and r.get("slab") is not None:
                m1_default.append({
                    "label": r["label"],
                    "miller_str": str(r["miller"]),
                    "E_slab_ev": r["energy_ev"],
                    "use": True,
                })
        if not m1_default:
            m1_default = [{"label": "example", "miller_str": "(1, 1, 1)", "E_slab_ev": -120.5, "use": False}]

        m1_df = st.data_editor(
            pd.DataFrame(m1_default),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "label": "Slab label",
                "miller_str": "Miller",
                "E_slab_ev": st.column_config.NumberColumn("E_slab (eV)", format="%.5f"),
                "use": st.column_config.CheckboxColumn("Calculate"),
            },
            key="m1_df",
        )

        run_m1 = st.button("Run Method 1", type="primary", key="run_m1")
        if run_m1:
            if not elemental_refs:
                st.error("Enter elemental reference energies above first.")
            elif not bulk_stoich:
                st.error("Enter bulk stoichiometry above first.")
            else:
                results_m1 = []
                errors_m1 = []
                for _, row in m1_df.iterrows():
                    if not row.get("use", False):
                        continue
                    # Find slab object and miller index from relaxed_slabs
                    slab_obj = None
                    slab_miller = None
                    for r in st.session_state.get("relaxed_slabs", []):
                        if r.get("label") == row["label"] and r.get("slab") is not None:
                            slab_obj = r["slab"]
                            slab_miller = r.get("miller")
                            break
                    if slab_obj is None:
                        errors_m1.append(f"{row['label']}: slab object not found (run MLFF relaxation first)")
                        continue
                    try:
                        res = se_mod.calc_surface_energy_method1(
                            slab=slab_obj,
                            slab_energy_ev=float(row["E_slab_ev"]),
                            bulk_comp_energy_ev=bulk_e_fu,
                            bulk_stoich=bulk_stoich,
                            elemental_refs_ev=elemental_refs,
                            normalizing_element=m1_norm_elem or None,
                            miller=slab_miller,
                        )
                        results_m1.append(res)
                    except Exception as e:
                        errors_m1.append(f"{row['label']}: {e}")

                if errors_m1:
                    for err in errors_m1:
                        st.warning(err)
                if results_m1:
                    st.session_state["surface_energies_m1"] = results_m1
                    rows_out = [
                        {
                            "Miller": str(r.miller),
                            "γ (eV/Å²)": f"{r.energy_ev_ang2:.6f}",
                            "γ (J/m²)": f"{r.energy_j_m2:.4f}",
                            "Method": r.method,
                        }
                        for r in results_m1
                    ]
                    st.success(f"Calculated {len(results_m1)} surface energies.")
                    st.dataframe(pd.DataFrame(rows_out), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # METHOD 2 — Formation energy reference
    # ─────────────────────────────────────────────────────────────────
    with st.expander("Method 2 — Bulk-like reference based on formation energy"):
        m2_left, m2_right = st.columns([1, 1])
        with m2_left:
            st.markdown("**Formula:**")
            st.latex(r"E_{\rm ref}(N) = \sum_i a_i \mu_i^\circ + N \cdot E_{\rm form}")
            st.latex(r"\gamma = \frac{E_{\rm slab} - E_{\rm ref}}{2A}")
            st.markdown(r"""
**Notation:**
- $a_i$ = atom count of element $i$ in slab
- $N = \sum_i a_i$ = total slab atoms
- $\mu_i^\circ$ = elemental reference energy (eV/atom)
- $E_{\rm form}$ = bulk formation energy per atom (eV/atom)
- $A$ = surface area (Å²)

**When to use:** Off-stoichiometric terminations; more general than Method 1.
            """)

        with m2_right:
            st.caption("Slab objects from Tab ④ relaxation are required.")
        m2_df = st.data_editor(
            pd.DataFrame(
                [{"label": r["label"], "E_slab_ev": r["energy_ev"], "use": True}
                 for r in st.session_state.get("relaxed_slabs", [])
                 if r.get("energy_ev") is not None]
                or [{"label": "example", "E_slab_ev": -120.5, "use": False}]
            ),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "label": "Slab label",
                "E_slab_ev": st.column_config.NumberColumn("E_slab (eV)", format="%.5f"),
                "use": st.column_config.CheckboxColumn("Calculate"),
            },
            key="m2_df",
        )

        run_m2 = st.button("Run Method 2", type="primary", key="run_m2")
        if run_m2:
            if not elemental_refs:
                st.error("Enter elemental reference energies above first.")
            else:
                results_m2 = []
                errors_m2 = []
                for _, row in m2_df.iterrows():
                    if not row.get("use", False):
                        continue
                    slab_obj = None
                    slab_miller = None
                    for r in st.session_state.get("relaxed_slabs", []):
                        if r.get("label") == row["label"] and r.get("slab") is not None:
                            slab_obj = r["slab"]
                            slab_miller = r.get("miller")
                            break
                    if slab_obj is None:
                        errors_m2.append(f"{row['label']}: slab object not found")
                        continue
                    try:
                        res = se_mod.calc_surface_energy_method2(
                            slab=slab_obj,
                            slab_energy_ev=float(row["E_slab_ev"]),
                            elemental_refs_ev=elemental_refs,
                            bulk_formation_energy_per_atom_ev=e_form_per_atom,
                            miller=slab_miller,
                        )
                        results_m2.append(res)
                    except Exception as e:
                        errors_m2.append(f"{row['label']}: {e}")

                for err in errors_m2:
                    st.warning(err)
                if results_m2:
                    st.session_state["surface_energies_m2"] = results_m2
                    rows_out = [
                        {"Miller": str(r.miller), "γ (eV/Å²)": f"{r.energy_ev_ang2:.6f}",
                         "γ (J/m²)": f"{r.energy_j_m2:.4f}", "Method": r.method}
                        for r in results_m2
                    ]
                    st.success(f"Calculated {len(results_m2)} surface energies.")
                    st.dataframe(pd.DataFrame(rows_out), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # METHOD 3 — Unified linear model / N-limit
    # ─────────────────────────────────────────────────────────────────
    with st.expander("Method 3 — Unified linear model (N-limit extrapolation)"):
        m3_left, m3_right = st.columns([1, 1])
        with m3_left:
            m3_mode = st.radio(
                "Method 3 mode",
                ["Direct fit (no references needed)", "With references (δE vs N)"],
                horizontal=False,
                key="m3_mode",
            )
        with m3_right:
            if m3_mode == "Direct fit (no references needed)":
                st.markdown("**Direct linear fit (no references):**")
                st.latex(r"E_{\rm slab}(N) = \varepsilon_{\rm bulk} \cdot N + 2\gamma A")
                st.markdown(r"""
Fit $E_{\rm slab}$ vs $N$ directly:
- slope $\rightarrow$ $\varepsilon_{\rm bulk}$ (bulk energy/atom, self-determined)
- intercept $\rightarrow$ $2\gamma A$, so $\gamma = \text{intercept}/(2A)$

**Advantage:** No external references needed.
                """)
            else:
                st.markdown("**N-limit extrapolation with references:**")
                st.latex(r"\delta E(N) = E_{\rm slab}(N) - E_{\rm ref}(N)")
                st.latex(r"E_{\rm ref}(N) = \sum_i a_i \mu_i^\circ + N \cdot E_{\rm form}")
                st.markdown(r"""
Fit $\delta E$ vs $N$: $\gamma = \text{intercept}/(2A)$

Linearity only in thick-slab regime — use convergence plot to identify converged region.
                """)

        st.divider()

        # ── Thickness series generator ──
        st.subheader("Generate thickness series")
        if st.session_state["bulk"] is None:
            st.warning("Load a bulk structure in Tab ① first.")
        else:
            gen_col1, gen_col2, gen_col3 = st.columns(3)
            with gen_col1:
                # Pick Miller index from generated slabs or enter manually
                available_millers = sorted({
                    str(s["miller"]) for s in st.session_state.get("slabs", [])
                    if s.get("error") is None
                })
                _all_opt = "— All —"
                if available_millers:
                    m3_miller_choices = [_all_opt] + available_millers
                    m3_miller_sel_list = st.multiselect(
                        "Miller indices (from Tab ②)",
                        m3_miller_choices,
                        default=[available_millers[0]] if available_millers else [],
                        key="m3_miller_sel",
                        help="Select one or more Miller indices, or '— All —' for all.",
                    )
                    # Expand "All" option
                    if _all_opt in m3_miller_sel_list:
                        m3_miller_sel_list = available_millers
                else:
                    _manual = st.text_input("Miller indices (comma-separated)", value="(1, 1, 1)", key="m3_miller_sel")
                    m3_miller_sel_list = [v.strip() for v in _manual.split(";") if v.strip()]
            with gen_col2:
                m3_thick_min = st.number_input("Min slab thickness (Å)", value=8.0, min_value=4.0, step=2.0, key="m3_thick_min")
                m3_thick_max = st.number_input("Max slab thickness (Å)", value=24.0, min_value=6.0, step=2.0, key="m3_thick_max")
                m3_thick_step = st.number_input("Step (Å)", value=4.0, min_value=1.0, step=1.0, key="m3_thick_step")
            with gen_col3:
                m3_vacuum = st.number_input("Vacuum (Å)", value=15.0, min_value=5.0, step=1.0, key="m3_vacuum")
                m3_gen_backend = st.selectbox(
                    "Relaxation backend",
                    ["MLFF (fairchem UMA)", "MLFF (local OCP checkpoint)", "Skip (energies only)"],
                    index=["MLFF (fairchem UMA)", "MLFF (local OCP checkpoint)", "Skip (energies only)"].index(
                        st.session_state.get("ref_backend", "MLFF (fairchem UMA)")
                    ) if st.session_state.get("ref_backend") in ["MLFF (fairchem UMA)", "MLFF (local OCP checkpoint)"] else 0,
                    key="m3_gen_backend",
                )
                if m3_gen_backend == "MLFF (fairchem UMA)":
                    _, _, m3_opt_mod, *_ = _import_core()
                    m3_gen_model = st.selectbox("UMA model", m3_opt_mod.AVAILABLE_UMA_MODELS,
                                                index=m3_opt_mod.AVAILABLE_UMA_MODELS.index(st.session_state.get("ref_model","uma-s-1"))
                                                if st.session_state.get("ref_model") in m3_opt_mod.AVAILABLE_UMA_MODELS else 0,
                                                key="m3_gen_model")
                    m3_gen_task = st.selectbox("Task", ["oc20","omat","omol"], key="m3_gen_task")
                elif m3_gen_backend == "MLFF (local OCP checkpoint)":
                    _, _, m3_opt_mod, *_ = _import_core()
                    m3_gen_ckpt = st.text_input("Checkpoint path", value=st.session_state.get("ref_checkpoint",""), key="m3_gen_ckpt")
                    m3_gen_cpu = st.checkbox("CPU", value=True, key="m3_gen_cpu")

            gen_series_btn = st.button("Generate & Relax thickness series", type="primary", key="gen_series_btn",
                                       disabled=st.session_state["bulk"] is None)
            if gen_series_btn:
                import re as _re2
                thicknesses = np.arange(m3_thick_min, m3_thick_max + 0.1, m3_thick_step).tolist()

                _, sg_mod_m3, opt_mod_m3, *_ = _import_core()
                from pymatgen.core.surface import SlabGenerator
                from pymatgen.io.ase import AseAtomsAdaptor as _AseAdaptor

                # Parse selected Miller indices
                hkl_list = []
                for _ms in m3_miller_sel_list:
                    _nums = [int(x) for x in _re2.findall(r"-?\d+", str(_ms))]
                    if len(_nums) >= 3:
                        hkl_list.append(tuple(_nums[:3]))
                if not hkl_list:
                    hkl_list = [(1, 1, 1)]

                series_rows = []
                total_steps = len(hkl_list) * len(thicknesses)
                prog = st.progress(0)
                step_i = 0
                for hkl_gen in hkl_list:
                    st.write(f"**Miller {hkl_gen}**")
                    for ti, thick in enumerate(thicknesses):
                        st_ph = st.empty()
                        st_ph.text(f"  {hkl_gen}  thickness {thick:.1f} Å — generating slab...")
                        try:
                            sg_t = SlabGenerator(
                                st.session_state["bulk"], hkl_gen,
                                min_slab_size=thick, min_vacuum_size=m3_vacuum,
                                primitive=True, center_slab=True,
                            )
                            slabs_t = sg_t.get_slabs(tol=0.1, symmetrize=True)
                            if not slabs_t:
                                slabs_t = sg_t.get_slabs(tol=0.05, symmetrize=False)
                            if not slabs_t:
                                st_ph.warning(f"  {hkl_gen} thickness {thick:.1f} Å: no slab generated, skipping")
                                step_i += 1
                                prog.progress(step_i / total_steps)
                                continue
                            slab_t = slabs_t[0]
                            atoms_t = _AseAdaptor().get_atoms(slab_t)
                            slab_dict_t = {
                                "label": f"{hkl_gen} t={thick:.0f}A",
                                "miller": hkl_gen,
                                "slab": slab_t,
                                "atoms": atoms_t,
                            }

                            if m3_gen_backend == "Skip (energies only)":
                                series_rows.append({
                                    "N_atoms": len(slab_t),
                                    "E_slab_ev": 0.0,
                                    "miller_str": str(hkl_gen),
                                    "use": True,
                                    "_note": f"t={thick:.0f}Å — energy not computed",
                                })
                                st_ph.text(f"  {hkl_gen} t={thick:.0f}Å  N={len(slab_t)} — skipped relaxation")
                            else:
                                st_ph.text(f"  {hkl_gen} t={thick:.0f}Å  N={len(slab_t)} — relaxing...")
                                if m3_gen_backend == "MLFF (fairchem UMA)":
                                    r_t = opt_mod_m3.relax_mlff(
                                        slab_dict_t, model_name=m3_gen_model,
                                        fmax=0.05, steps=200, task_name=m3_gen_task,
                                    )
                                else:
                                    r_t = opt_mod_m3.relax_ocp_checkpoint(
                                        slab_dict_t, checkpoint_path=m3_gen_ckpt,
                                        fmax=0.05, steps=200, cpu=m3_gen_cpu,
                                    )
                                if r_t.get("error"):
                                    st_ph.warning(f"  {hkl_gen} t={thick:.0f}Å: {r_t['error'][:100]}")
                                else:
                                    series_rows.append({
                                        "N_atoms": len(slab_t),
                                        "E_slab_ev": r_t["energy_ev"],
                                        "miller_str": str(hkl_gen),
                                        "use": True,
                                    })
                                    st_ph.text(
                                        f"  {hkl_gen} t={thick:.0f}Å  N={len(slab_t)}  "
                                        f"E={r_t['energy_ev']:.4f} eV  "
                                        f"fmax={r_t['fmax_final']:.4f}  conv={r_t['converged']}"
                                    )
                        except Exception as ex:
                            st_ph.warning(f"  {hkl_gen} t={thick:.0f}Å failed: {ex}")
                        step_i += 1
                        prog.progress(step_i / total_steps)

                if series_rows:
                    st.session_state["m3_thickness_series"] = series_rows
                    st.success(f"Generated {len(series_rows)} thickness points — table updated below.")

        st.divider()
        st.write("**Slab series data** (auto-filled from generation above, or enter manually):")
        st.caption("Each row = one slab thickness. Same Miller index and termination shift.")

        # Use generated series if available, else defaults
        _m3_stored = st.session_state.get("m3_thickness_series")
        m3_rows_default = _m3_stored if _m3_stored else [
            {"N_atoms": n, "E_slab_ev": e, "miller_str": "(1,1,1)", "use": True}
            for n, e in [(20, -80.0), (30, -120.0), (40, -160.1), (50, -200.2), (60, -240.2)]
        ]
        m3_df = st.data_editor(
            pd.DataFrame(m3_rows_default),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "N_atoms": st.column_config.NumberColumn("N atoms", format="%d"),
                "E_slab_ev": st.column_config.NumberColumn("E_slab (eV)", format="%.5f"),
                "miller_str": "Miller index",
                "use": st.column_config.CheckboxColumn("Include"),
            },
            key="m3_df",
        )

        m3_area_col, m3_r2_col = st.columns(2)
        m3_area = m3_area_col.number_input("Surface area A (Å²)", value=15.0, min_value=1.0, step=0.5, key="m3_area",
                                            help="Auto-compute: |a×b| of slab cell. For Cu(111) primitive ≈ 5.6 Å².")
        m3_min_r2 = m3_r2_col.slider("Min R² for convergence flag", 0.90, 1.0, 0.999, 0.001, key="m3_r2")

        m3_col1, m3_col2 = st.columns(2)
        run_m3_conv = m3_col1.button("Show convergence plot", key="run_m3_conv")
        run_m3 = m3_col2.button("Run Method 3 (N-limit fit)", type="primary", key="run_m3")

        # Convergence plot
        if run_m3_conv:
            if m3_mode == "Direct fit (no references needed)":
                try:
                    from core.surface_energy import convergence_plot_data_direct, EV_ANG2_TO_J_M2 as _ev2j
                    used_rows = m3_df[m3_df["use"] == True]
                    n_list = used_rows["N_atoms"].tolist()
                    e_list = used_rows["E_slab_ev"].tolist()
                    conv = convergence_plot_data_direct(n_list, e_list, m3_area)

                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=conv["n_atoms"][1:],  # running fit starts at point 2
                        y=conv["gamma_running_j_m2"],
                        mode="lines+markers", name="γ (running fit)",
                        line=dict(color="#1f77b4", width=2), marker=dict(size=8),
                    ))
                    fig_conv.add_hline(
                        y=conv["intercept"] / (2.0 * m3_area) * 16.0218,
                        line_dash="dash",
                        annotation_text=f"Final γ = {conv['intercept']/(2*m3_area)*16.0218:.4f} J/m²",
                    )
                    fig_conv.update_layout(
                        title="Convergence: γ(N) from running linear fit",
                        xaxis_title="N atoms included in fit",
                        yaxis_title="γ (J/m²)", height=380,
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)
                except Exception as e:
                    st.error(f"Convergence plot failed: {e}")
            else:
                if not elemental_refs:
                    st.error("Enter elemental reference energies above first.")
                else:
                    try:
                        from core.surface_energy import convergence_plot_data, EV_ANG2_TO_J_M2 as _ev2j
                        used_rows = m3_df[m3_df["use"] == True]
                        n_list = used_rows["N_atoms"].tolist()
                        e_list = used_rows["E_slab_ev"].tolist()

                        conv_data = convergence_plot_data(
                            slabs=None,  # not needed for this path
                            slab_energies_ev=e_list,
                            elemental_refs_ev=elemental_refs,
                            bulk_formation_energy_per_atom_ev=e_form_per_atom,
                            area_ang2=m3_area,
                            _n_atoms_override=n_list,
                        )
                        fig_conv = go.Figure()
                        fig_conv.add_trace(go.Scatter(
                            x=conv_data["n_atoms"],
                            y=[g * _ev2j for g in conv_data["gamma_raw_ev_ang2"]],
                            mode="lines+markers",
                            name="γ(N)",
                            line=dict(color="#1f77b4", width=2),
                            marker=dict(size=8),
                        ))
                        fig_conv.update_layout(
                            title="Convergence: γ(N) vs Atom Count  [Fig. 1 style]",
                            xaxis_title="N atoms in slab",
                            yaxis_title="γ (J/m²)",
                            height=380,
                        )
                        st.plotly_chart(fig_conv, use_container_width=True)
                    except Exception as e:
                        st.error(f"Convergence plot failed: {e}")

        if run_m3:
            if m3_mode == "Direct fit (no references needed)":
                try:
                    from core.surface_energy import calc_surface_energy_method3_direct, convergence_plot_data_direct
                    used = m3_df[m3_df["use"] == True]
                    n_list = used["N_atoms"].astype(int).tolist()
                    e_list = used["E_slab_ev"].astype(float).tolist()
                    miller_str = used["miller_str"].iloc[0] if len(used) > 0 else "(1,1,1)"
                    import re as _re
                    hkl_nums = [int(x) for x in _re.findall(r"-?\d+", miller_str)]
                    hkl = tuple(hkl_nums[:3]) if len(hkl_nums) >= 3 else (1, 1, 1)

                    res_m3 = calc_surface_energy_method3_direct(n_list, e_list, m3_area, hkl, m3_min_r2)

                    # Show fit plot using E_slab vs N_atoms
                    conv = convergence_plot_data_direct(n_list, e_list, m3_area)
                    fig_fit = go.Figure()
                    fig_fit.add_trace(go.Scatter(x=n_list, y=e_list, mode="markers", name="E_slab (eV)", marker=dict(size=10)))
                    fig_fit.add_trace(go.Scatter(x=conv["n_fit"], y=conv["e_fit"], mode="lines", name=f"fit (R²={res_m3.metadata['r2']:.5f})"))
                    fig_fit.update_layout(title="Method 3 (direct): E_slab vs N atoms", xaxis_title="N atoms", yaxis_title="E_slab (eV)", height=350)
                    st.plotly_chart(fig_fit, use_container_width=True)

                    converged_str = "✓ Converged" if res_m3.converged else f"✗ Not converged (R²={res_m3.metadata['r2']:.5f})"
                    st.success(f"γ = {res_m3.energy_ev_ang2:.6f} eV/Å² = {res_m3.energy_j_m2:.4f} J/m²  |  {converged_str}")
                    st.caption(f"Bulk energy from fit: {res_m3.metadata['slope_ev_atom']:.5f} eV/atom")

                    existing_m3 = st.session_state.get("surface_energies_m3", [])
                    existing_m3 = [r for r in existing_m3 if r.miller != hkl]
                    existing_m3.append(res_m3)
                    st.session_state["surface_energies_m3"] = existing_m3

                except Exception as e:
                    st.error(f"Method 3 (direct) failed: {e}")
                    st.code(traceback.format_exc())
            else:
                if not elemental_refs:
                    st.error("Enter elemental reference energies above first.")
                else:
                    try:
                        used = m3_df[m3_df["use"] == True]
                        n_list = used["N_atoms"].astype(int).tolist()
                        e_list = used["E_slab_ev"].astype(float).tolist()
                        miller_str = used["miller_str"].iloc[0] if len(used) > 0 else "(1,1,1)"

                        # Parse miller
                        import re as _re
                        hkl_nums = [int(x) for x in _re.findall(r"-?\d+", miller_str)]
                        hkl = tuple(hkl_nums[:3]) if len(hkl_nums) >= 3 else (1, 1, 1)

                        # Build mock slabs for the function (using composition proxy from N_atoms)
                        # Since we don't have real Slab objects here, use the direct formula
                        # δE(N) = E_slab(N) - [Σ a_i μ_i° + N × E_form]
                        # With known composition fractions from bulk_stoich
                        total_stoich = sum(bulk_stoich.values()) if bulk_stoich else 1.0
                        delta_e_list = []
                        for n_at, e_sl in zip(n_list, e_list):
                            if bulk_stoich and elemental_refs:
                                e_elem = sum(
                                    (bulk_stoich.get(el, 0) / total_stoich) * n_at * elemental_refs[el]
                                    for el in elemental_refs if el in bulk_stoich
                                )
                            else:
                                e_elem = 0.0
                            e_ref_n = e_elem + n_at * e_form_per_atom
                            delta_e_list.append(e_sl - e_ref_n)

                        # Linear fit: δE = b + c×N  →  γ = b / (2A)
                        n_arr = np.array(n_list, dtype=float)
                        de_arr = np.array(delta_e_list, dtype=float)
                        coeffs = np.polyfit(n_arr, de_arr, deg=1)
                        c_sl, b_int = coeffs
                        gamma_ev = b_int / (2.0 * m3_area)

                        de_pred = np.polyval(coeffs, n_arr)
                        ss_res = np.sum((de_arr - de_pred) ** 2)
                        ss_tot = np.sum((de_arr - np.mean(de_arr)) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

                        from core.surface_energy import EV_ANG2_TO_J_M2 as _ev2j, SurfaceEnergyResult
                        res_m3 = SurfaceEnergyResult.from_ev_ang2(
                            miller=hkl,
                            energy_ev_ang2=gamma_ev,
                            converged=(r2 >= m3_min_r2),
                            method="method3_nlimit",
                            metadata={"r2": r2, "intercept": b_int, "slope": c_sl, "area": m3_area},
                        )

                        converged_str = "✓ Converged" if res_m3.converged else f"✗ Not converged (R²={r2:.5f} < {m3_min_r2})"
                        st.success(f"γ = {res_m3.energy_ev_ang2:.6f} eV/Å² = {res_m3.energy_j_m2:.4f} J/m²  |  {converged_str}")

                        # Fit visualization
                        fig_fit = go.Figure()
                        fig_fit.add_trace(go.Scatter(x=n_list, y=delta_e_list, mode="markers", name="δE(N)", marker=dict(size=10)))
                        n_fine = np.linspace(min(n_list), max(n_list), 100)
                        fig_fit.add_trace(go.Scatter(x=n_fine, y=np.polyval(coeffs, n_fine), mode="lines", name=f"fit (R²={r2:.5f})"))
                        fig_fit.add_hline(y=b_int, line_dash="dash", annotation_text=f"intercept = 2γA = {b_int:.4f} eV")
                        fig_fit.update_layout(
                            title="Method 3: δE(N) = E_slab − E_ref vs N",
                            xaxis_title="N atoms", yaxis_title="δE (eV)", height=350,
                        )
                        st.plotly_chart(fig_fit, use_container_width=True)

                        # Save result
                        existing_m3 = st.session_state.get("surface_energies_m3", [])
                        existing_m3 = [r for r in existing_m3 if r.miller != hkl]
                        existing_m3.append(res_m3)
                        st.session_state["surface_energies_m3"] = existing_m3

                    except Exception as e:
                        st.error(f"Method 3 failed: {e}")
                        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7: WULFF CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Wulff Construction")

    if st.session_state["bulk"] is None:
        st.info("Load a bulk structure in Tab ① first.")
    else:
        _, _, _, _, wulff_mod, *_ = _import_core()

        # ── Energy source selector ──
        _avail_methods = []
        if st.session_state.get("surface_energies_m1"):
            _avail_methods.append("Method 1")
        if st.session_state.get("surface_energies_m2"):
            _avail_methods.append("Method 2")
        if st.session_state.get("surface_energies_m3"):
            _avail_methods.append("Method 3")
        if st.session_state.get("manual_energies"):
            _avail_methods.append("Manual input")

        if not _avail_methods:
            st.warning("No surface energies available. Run a calculation method in Tab ⑥ first.")
            st.stop()

        energy_source = st.radio(
            "Surface energy source",
            _avail_methods,
            horizontal=True,
        )

        if energy_source == "Method 1":
            results = st.session_state.get("surface_energies_m1", [])
            miller_energies = [(r.miller, r.energy_j_m2) for r in results]
        elif energy_source == "Method 2":
            results = st.session_state.get("surface_energies_m2", [])
            miller_energies = [(r.miller, r.energy_j_m2) for r in results]
        elif energy_source == "Method 3":
            results = st.session_state.get("surface_energies_m3", [])
            miller_energies = [(r.miller, r.energy_j_m2) for r in results]
        else:
            miller_energies = list(st.session_state.get("manual_energies", []))

        # Keep only the lowest-energy termination per Miller index;
        # drop NaN/inf/non-positive and invalid (0,0,0) Miller indices
        import math
        _best: dict[tuple, float] = {}
        _dropped_invalid = []
        for hkl, e in miller_energies:
            if sum(abs(x) for x in hkl) == 0:
                _dropped_invalid.append(f"{hkl} (zero Miller index)")
                continue
            if not math.isfinite(e) or e <= 0:
                _dropped_invalid.append(f"{hkl} (γ={e})")
                continue
            if hkl not in _best or e < _best[hkl]:
                _best[hkl] = e
        if _dropped_invalid:
            st.warning(
                f"Dropped {len(_dropped_invalid)} invalid facet(s): {_dropped_invalid}"
            )
        miller_energies = list(_best.items())

        if not miller_energies:
            st.warning("All facets were filtered out (invalid Miller index or non-positive energy). Check Tab ⑥ results.")
        else:
            # ── Controls row ──
            wulff_col_ctrl, wulff_col_sort = st.columns([2, 1])
            with wulff_col_ctrl:
                energy_unit = st.selectbox("Display energy unit", ["J/m²", "eV/Å²"])
            with wulff_col_sort:
                sort_mode = st.radio(
                    "Bar chart sort order",
                    ["energy_asc", "facet_name", "area_frac_desc"],
                    format_func={
                        "energy_asc": "Surface energy ↑",
                        "facet_name": "Facet name (A→Z)",
                        "area_frac_desc": "Area fraction ↓",
                    }.get,
                )

            # ── Manual color picker ──
            with st.expander("Custom facet colors", expanded=False):
                st.caption("Pick a color for each facet. Leave as default to use auto-assigned colors.")
                _default_colors = wulff_mod.build_color_map(
                    [hkl for hkl, _ in miller_energies], exposed_millers=None
                )
                custom_colors = {}
                color_cols = st.columns(min(len(miller_energies), 4))
                for idx, (hkl, _) in enumerate(miller_energies):
                    label = f"({hkl[0]} {hkl[1]} {hkl[2]})"
                    col = color_cols[idx % len(color_cols)]
                    picked = col.color_picker(
                        label,
                        value=_default_colors.get(hkl, "#AAAAAA"),
                        key=f"color_{idx}_{hkl[0]}_{hkl[1]}_{hkl[2]}",
                    )
                    custom_colors[hkl] = picked

            build_btn = st.button("Build Wulff Construction", type="primary")

            if build_btn:
                with st.spinner("Computing Wulff shape..."):
                    try:
                        disp_energies = list(miller_energies)
                        if energy_unit == "eV/Å²":
                            from core.surface_energy import EV_ANG2_TO_J_M2
                            disp_energies = [(hkl, e / EV_ANG2_TO_J_M2) for hkl, e in miller_energies]

                        wulff_shape, area_fractions = wulff_mod.compute_wulff(
                            st.session_state["bulk"],
                            miller_energies,
                        )

                        # Use custom colors if provided, merged with exposed info
                        exposed_set = {hkl for hkl, frac in area_fractions.items() if frac > 0.001}
                        color_map = dict(custom_colors)  # already per-facet from picker

                        wulff_fig = wulff_mod.wulff_3d_figure(
                            wulff_shape, area_fractions, color_map,
                        )
                        bar_fig = wulff_mod.bar_chart_figure(
                            disp_energies, area_fractions, color_map,
                            sort_mode=sort_mode, energy_unit=energy_unit,
                        )

                        # ── Metrics row ──
                        exposed = {hkl: frac for hkl, frac in area_fractions.items() if frac > 0.001}
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Exposed facets", len(exposed))
                        # Weighted average surface energy
                        w_avg = sum(
                            area_fractions.get(hkl, 0.0) * e
                            for hkl, e in miller_energies
                        )
                        m_col2.metric("Weighted avg γ", f"{w_avg:.4f} J/m²")
                        # Anisotropy = γ_max / γ_min (exposed only)
                        exposed_energies = [e for hkl, e in miller_energies if area_fractions.get(hkl, 0) > 0.001]
                        if len(exposed_energies) >= 2:
                            aniso = max(exposed_energies) / min(exposed_energies)
                            m_col3.metric("Anisotropy γ_max/γ_min", f"{aniso:.3f}")
                        # Total surface area (Wulff shape)
                        try:
                            m_col4.metric("Wulff surface area", f"{wulff_shape.surface_area:.4f} Å²")
                        except Exception:
                            pass

                        # ── Plots ──
                        fig_col1, fig_col2 = st.columns([1.2, 1])
                        with fig_col1:
                            st.plotly_chart(wulff_fig, use_container_width=True)
                        with fig_col2:
                            st.plotly_chart(bar_fig, use_container_width=True)

                        # ── Detailed parameters table ──
                        st.divider()
                        st.subheader("Wulff Construction Parameters")

                        af_rows = []
                        for hkl, energy in sorted(miller_energies, key=lambda x: -area_fractions.get(x[0], 0)):
                            frac = area_fractions.get(hkl, 0.0)
                            e_display = energy if energy_unit == "J/m²" else energy / 16.0218
                            e_ev = energy / 16.0218
                            af_rows.append({
                                "Facet": f"({hkl[0]} {hkl[1]} {hkl[2]})",
                                f"γ ({energy_unit})": round(e_display, 5),
                                "γ (eV/Å²)": round(e_ev, 6),
                                "γ (J/m²)": round(energy, 5),
                                "Area fraction (%)": round(frac * 100, 3),
                                "Exposed": "✓" if frac > 0.001 else "—",
                                "Color": custom_colors.get(hkl, "#AAAAAA"),
                            })
                        st.dataframe(pd.DataFrame(af_rows), use_container_width=True)

                        # ── Extra Wulff shape properties ──
                        st.subheader("Shape Properties")
                        prop_col1, prop_col2 = st.columns(2)
                        with prop_col1:
                            try:
                                st.write(f"**Volume:** {wulff_shape.volume:.5f} Å³")
                                st.write(f"**Surface area:** {wulff_shape.surface_area:.5f} Å²")
                                st.write(f"**Total facets (including unexposed):** {len(miller_energies)}")
                                st.write(f"**Exposed facets:** {len(exposed)}")
                            except Exception:
                                pass
                        with prop_col2:
                            try:
                                st.write(f"**Weighted avg surface energy:** {w_avg:.5f} J/m²")
                                if len(exposed_energies) >= 2:
                                    st.write(f"**Min γ (exposed):** {min(exposed_energies):.5f} J/m²")
                                    st.write(f"**Max γ (exposed):** {max(exposed_energies):.5f} J/m²")
                                    st.write(f"**Anisotropy γ_max/γ_min:** {aniso:.4f}")
                                # Sphericity: surface_area of equivalent sphere / actual surface_area
                                import math
                                r_sphere = (3 * wulff_shape.volume / (4 * math.pi)) ** (1/3)
                                a_sphere = 4 * math.pi * r_sphere ** 2
                                sphericity = a_sphere / wulff_shape.surface_area
                                st.write(f"**Sphericity:** {sphericity:.4f}  (1.0 = perfect sphere)")
                            except Exception:
                                pass

                    except Exception as e:
                        st.error(f"Wulff construction failed: {e}")
                        st.code(traceback.format_exc())
