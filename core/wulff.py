"""
wulff.py
Wulff construction computation and plotly figure builders.
Adapted from wulff_plot.py (matplotlib) to plotly for in-browser rendering.
"""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.core import Structure
from pymatgen.analysis.wulff import WulffShape

from .surface_energy import SurfaceEnergyResult


# ─────────────────────────────────────────────────────────────────────────────
# Color utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def build_color_map(
    miller_list: list[tuple[int, int, int]],
    exposed_millers: set[tuple[int, int, int]] | None = None,
) -> dict[tuple, str]:
    """
    Assign a distinct hex color to each Miller index.
    Unexposed facets get a muted gray.
    """
    n = len(miller_list)
    color_map = {}
    for i, hkl in enumerate(miller_list):
        if exposed_millers is not None and hkl not in exposed_millers:
            color_map[hkl] = "#AAAAAA"
        else:
            h = i / max(n, 1)
            color_map[hkl] = _hsl_to_hex(h, 0.75, 0.55)
    return color_map


# ─────────────────────────────────────────────────────────────────────────────
# Core Wulff construction
# ─────────────────────────────────────────────────────────────────────────────

def compute_wulff(
    structure: Structure,
    miller_energies: list[tuple[tuple[int, int, int], float]],
) -> tuple[WulffShape, dict[tuple, float]]:
    """
    Build a pymatgen WulffShape from structure lattice and surface energies.

    Parameters
    ----------
    structure       : Bulk crystal structure (provides lattice)
    miller_energies : List of ((h,k,l), energy_j_m2) tuples

    Returns
    -------
    (wulff_shape, area_fraction_dict)
        area_fraction_dict maps (h,k,l) -> fractional area on Wulff shape
    """
    miller_list = [hkl for hkl, _ in miller_energies]
    energy_list = [e for _, e in miller_energies]

    wulff = WulffShape(
        lattice=structure.lattice,
        miller_list=miller_list,
        e_surf_list=energy_list,
    )

    # area_fraction_dict is a native attribute: {(h,k,l): float}
    area_fractions = {
        tuple(hkl): float(frac)
        for hkl, frac in wulff.area_fraction_dict.items()
    }

    return wulff, area_fractions


def compute_wulff_from_results(
    structure: Structure,
    results: list[SurfaceEnergyResult],
) -> tuple[WulffShape, dict[tuple, float]]:
    """
    Convenience wrapper: build Wulff from SurfaceEnergyResult list.
    Uses energy_j_m2 from each result.
    """
    miller_energies = [(r.miller, r.energy_j_m2) for r in results]
    return compute_wulff(structure, miller_energies)


# ─────────────────────────────────────────────────────────────────────────────
# 3D Wulff shape figure
# ─────────────────────────────────────────────────────────────────────────────

def wulff_3d_figure(
    wulff_shape: WulffShape,
    area_fractions: dict[tuple, float],
    color_map: dict[tuple, str],
    show_miller_labels: bool = True,
    opacity: float = 0.85,
) -> go.Figure:
    """
    Build a plotly Figure with go.Mesh3d for the Wulff shape.
    Each facet gets its own Mesh3d trace (for independent coloring and legend).

    Returns a Figure ready for st.plotly_chart().
    """
    traces = []

    for facet in wulff_shape.facets:
        hkl = tuple(facet.miller)
        area_frac = area_fractions.get(hkl, 0.0)

        # Skip unexposed facets (no triangles)
        if not facet.points:
            continue

        color = color_map.get(hkl, "#AAAAAA")
        label = f"({hkl[0]} {hkl[1]} {hkl[2]})"

        # facet.points is a list of triangles, each triangle = [v0, v1, v2]
        # where each v is a numpy array of shape (3,)
        all_x, all_y, all_z = [], [], []
        i_idx, j_idx, k_idx = [], [], []

        for tri in facet.points:
            tri = np.array(tri)  # shape (3, 3)
            if tri.shape != (3, 3):
                continue
            base = len(all_x)
            all_x.extend(tri[:, 0].tolist())
            all_y.extend(tri[:, 1].tolist())
            all_z.extend(tri[:, 2].tolist())
            i_idx.append(base)
            j_idx.append(base + 1)
            k_idx.append(base + 2)

        if not all_x:
            continue

        x, y, z = all_x, all_y, all_z
        n = len(x)

        hover_text = (
            f"<b>{label}</b><br>"
            f"Area fraction: {area_frac*100:.1f}%<br>"
            f"Vertices: {n}"
        )

        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=i_idx, j=j_idx, k=k_idx,
            color=color,
            opacity=opacity,
            name=label,
            legendgroup=label,
            showlegend=True,
            hovertemplate=hover_text + "<extra></extra>",
            flatshading=False,
            lighting=dict(diffuse=0.8, specular=0.3, ambient=0.3),
            lightposition=dict(x=100, y=200, z=0),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            title="Facets",
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        title=dict(text="Wulff Shape", x=0.5),
        height=550,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bar chart: facet vs surface energy + area fraction annotations
# ─────────────────────────────────────────────────────────────────────────────

def bar_chart_figure(
    miller_energies: list[tuple[tuple[int, int, int], float]],
    area_fractions: dict[tuple, float],
    color_map: dict[tuple, str],
    sort_mode: str = "energy_asc",
    energy_unit: str = "J/m²",
) -> go.Figure:
    """
    Build a bar chart: x=facet label, y=surface energy, annotated with area %.

    Parameters
    ----------
    miller_energies : [(hkl, energy), ...] — energies in J/m²
    area_fractions  : {hkl: fraction} from compute_wulff()
    color_map       : {hkl: hex_color}
    sort_mode       : One of:
                        "facet_name"      — alphabetical by (h k l) string
                        "energy_asc"      — surface energy low → high
                        "area_frac_desc"  — area fraction high → low
    energy_unit     : Label for y-axis ("J/m²" or "eV/Å²")

    Returns
    -------
    plotly Figure
    """
    # Sorting
    data = list(miller_energies)

    if sort_mode == "facet_name":
        data.sort(key=lambda x: str(x[0]))
    elif sort_mode == "energy_asc":
        data.sort(key=lambda x: x[1])
    elif sort_mode == "area_frac_desc":
        data.sort(key=lambda x: -area_fractions.get(x[0], 0.0))
    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode!r}")

    labels = [f"({h[0][0]} {h[0][1]} {h[0][2]})" for h in data]
    energies = [h[1] for h in data]
    colors = [color_map.get(h[0], "#AAAAAA") for h in data]
    fractions = [area_fractions.get(h[0], 0.0) for h in data]

    # Annotation texts: area fraction above each bar
    annotations = []
    for i, (label, energy, frac) in enumerate(zip(labels, energies, fractions)):
        if frac > 0.001:
            ann_text = f"{frac*100:.1f}%"
        else:
            ann_text = "—"  # Not exposed on Wulff shape
        annotations.append(dict(
            x=i,
            y=energy,
            text=ann_text,
            showarrow=False,
            yshift=8,
            font=dict(size=10, color="black"),
            xref="x",
            yref="y",
        ))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=energies,
        marker_color=colors,
        marker_line=dict(width=0.8, color="black"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"Surface energy: %{{y:.3f}} {energy_unit}<br>"
            "Area fraction: %{customdata:.1f}%<extra></extra>"
        ),
        customdata=[f * 100 for f in fractions],
        name="Surface energy",
    ))

    fig.update_layout(
        title=dict(text="Surface Energies & Wulff Area Fractions", x=0.5),
        xaxis=dict(title="Facet (h k l)", tickangle=-30),
        yaxis=dict(title=f"Surface energy ({energy_unit})"),
        annotations=annotations,
        showlegend=False,
        height=420,
        margin=dict(l=60, r=20, t=50, b=80),
        bargap=0.25,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Combined figure (3D + bar chart, side by side)
# ─────────────────────────────────────────────────────────────────────────────

def combined_figure(
    wulff_shape: WulffShape,
    area_fractions: dict[tuple, float],
    miller_energies: list[tuple[tuple, float]],
    sort_mode: str = "energy_asc",
    energy_unit: str = "J/m²",
) -> tuple[go.Figure, go.Figure]:
    """
    Returns (wulff_3d_fig, bar_fig) as separate figures.
    (Streamlit renders them side-by-side with st.columns.)
    """
    miller_list = [hkl for hkl, _ in miller_energies]
    exposed = set(area_fractions.keys())
    color_map = build_color_map(miller_list, exposed)

    wulff_fig = wulff_3d_figure(wulff_shape, area_fractions, color_map)
    bar_fig = bar_chart_figure(miller_energies, area_fractions, color_map, sort_mode, energy_unit)

    return wulff_fig, bar_fig
