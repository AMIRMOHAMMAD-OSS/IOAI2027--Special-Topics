#!/usr/bin/env python3
"""Regenerate original figures from the checked, machine-readable reports."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REF = ROOT / "results" / "reference"
FIG.mkdir(exist_ok=True)
INK, MUTED, PAPER = "#162c3c", "#546b79", "#fafaf6"
PURPLE, TEAL, ORANGE = "#6943ae", "#087f8c", "#d17b2e"
COLORS = ["#168c9c", "#cf6b3a", "#7560b3", "#b44972"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#b0bec5",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER, "svg.fonttype": "none"})


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def profiles():
    p = json.loads((REF / "sequential_profile.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    fig.subplots_adjust(top=.76, bottom=.20, wspace=.14)
    fig.text(.07, .94, "Local steps. Collective constraints.", fontsize=24, weight="bold")
    fig.text(.07, .875, "Four-ring propeller chain · independent MMFF94 model · common energy reference A", color=MUTED)
    x, y = np.array(p["normalized_arc_length"]), np.array(p["energy_above_A_kcal_mol"])
    axes[0].plot(x, y, color=PURPLE, lw=2.4)
    aliases = {"N002": "P", "N015": "Q"}
    for label, point in p["markers"].items():
        if not label.startswith(("TS", "C_pair", "rescue", "R_N")):
            axes[0].scatter(point["progress"], point["energy_above_A_kcal_mol"], s=29, color=PURPLE, zorder=3)
            axes[0].annotate(aliases.get(label, label), (point["progress"], point["energy_above_A_kcal_mol"]),
                             xytext=(0, -18), textcoords="offset points", ha="center", weight="bold", color=PURPLE)
    axes[0].text(.03, 15.7, "7.404 kcal/mol", fontsize=21, color=PURPLE, weight="bold")
    axes[0].text(.03, 14.2, "Highest of six checked saddles", color=MUTED, fontsize=10)
    axes[0].set_title("Validated localized route", loc="left", fontsize=13, pad=14, weight="bold")
    axes[0].set_xlabel("Normalized recorded path length\n(not time or collective progress)")
    axes[0].set_ylabel("Potential energy above A  [kcal/mol]")
    collective = REF / "collective_profiles.json"
    if collective.exists():
        series = json.loads(collective.read_text())
        for i, branch in enumerate(series["branches"]):
            color = TEAL if branch["pattern"] == "I_III" else ORANGE
            t = np.array(branch["progress"])
            energy = np.array(branch["energy_above_A_kcal_mol"])
            marker = "o" if branch["direction"] == "forward" else "+"
            axes[1].scatter(t, energy, s=10, marker=marker, color=color, alpha=.20)
        for pattern, color in (("I_III", TEAL), ("II_IV", ORANGE)):
            with np.load(REF / (pattern + "_complete_display.npz")) as z:
                axes[1].plot(z["progress"], z["energy_above_A_kcal_mol"], color=color, lw=2.0,
                    label=pattern.replace("_", "/") + f"  {z['energy_above_A_kcal_mol'].max():.3f} kcal/mol")
        axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_title("Checked synchronized candidates", loc="left", fontsize=13, pad=14, weight="bold")
    axes[1].set_xlabel("Prescribed six-torsion progress, t\n(fixed-progress conformational connectors included)")
    for ax in axes:
        ax.set_ylim(-1.2, 18)
        ax.set_xlim(-.025, 1.025)
        ax.axhline(7.40399469869, color=PURPLE, ls="--", lw=1, alpha=.7)
        ax.grid(axis="y", alpha=.22)
    fig.text(.07, .06, "Explicit candidate paths in MMFF94. Potential energies, finite path checks, and no claim of global optimality.", fontsize=10, color=MUTED)
    save(fig, "energy_profiles")


def activity():
    d = json.loads((REF / "mechanism_geometry.json").read_text())
    rows = d["events"]
    data = 100 * np.array([r["pair_squared_torsion_derivative_fractions"] for r in rows])
    fig, ax = plt.subplots(figsize=(10.6, 5.1))
    fig.subplots_adjust(top=.75, bottom=.22, left=.20, right=.83)
    cmap = LinearSegmentedColormap.from_list("activity", ["#f2eef8", PURPLE])
    image = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(6):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center",
                    color="white" if data[i,j] > 60 else INK, fontsize=11)
    ax.set_xticks(range(3), ["I–II", "II–III", "III–IV"])
    ax.set_yticks(range(6), ["A → C", "C → P", "P → Q", "Q → D", "D → M*", "M → B"])
    ax.set_xlabel("Adjacent ring pair in the fixed path atom frame", labelpad=10)
    fig.colorbar(image, ax=ax, fraction=.045, pad=.05, label="Share of squared six-torsion derivative [%]")
    fig.text(.07, .92, "Five localized disrotations", fontsize=23, weight="bold")
    fig.text(.07, .845, "Unstable-mode torsion activity identifies which adjacent ring pair moves at each saddle.", color=MUTED)
    fig.text(.07, .075, "* D → M is a methyl/conformational adjustment. Its small ring signal is normalized here;\n  the largest methyl derivative is 39.09°/Å versus a six-ring-torsion norm of 5.73°/Å.", fontsize=10, color=MUTED)
    save(fig, "torsion_activity")


def convergence():
    path = REF / "collective_profiles.json"
    if not path.exists():
        return
    profiles = json.loads(path.read_text())["branches"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    fig.subplots_adjust(top=.72, bottom=.23, wspace=.35)
    fig.text(.07, .94, "Convergence is measured, not assumed", fontsize=23, weight="bold")
    fig.text(.07, .86, "Every accepted node passes angular, stationarity, and constrained-curvature checks.", color=MUTED)
    fields = ["max_constraint_rad", "kkt_gradient_max_atomic_norm", "lowest_tangent_curvature"]
    labels = ["Maximum angular residual [rad]", "KKT force [kcal/(mol Å)]", "Lowest constrained curvature"]
    for branch in profiles:
        color = TEAL if branch["pattern"] == "I_III" else ORANGE
        marker = "o" if branch["direction"] == "forward" else "+"
        for ax, key in zip(axes, fields):
            y = np.maximum(np.asarray(branch[key]), 1e-16)
            ax.scatter(branch["progress"], y, s=15, marker=marker, color=color, alpha=.7)
    for ax, label in zip(axes, labels):
        ax.set_xlabel("Collective progress, t")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=.2)
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[0].axhline(1e-10, color=INK, ls="--", lw=.8)
    axes[1].axhline(1e-7, color=INK, ls="--", lw=.8)
    axes[2].axhline(0, color=INK, lw=.8)
    fig.text(.07, .055, "Curvature uses the Hessian of the Lagrangian projected onto the 285-dimensional constraint tangent space.", color=MUTED, fontsize=10)
    save(fig, "convergence")


def network():
    path = REF / "network.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    nodes, edges = data["nodes"], data["edges"]
    reachable = set(data["coverage"]["reachable_node_ids"])
    names = [name for name in nodes if name in reachable]
    selected = ["A", "C", "N002", "N015", "D", "M", "B"]
    idx = {name: i for i, name in enumerate(names)}
    rng = np.random.default_rng(1993)
    xy = rng.normal(0, 1, (len(names), 2))
    pairs = [(idx[e["endpoints"][0]], idx[e["endpoints"][1]]) for e in edges
             if set(e["endpoints"]).issubset(reachable)]
    anchors = np.array([[-3.,0.],[-2.,1.6],[-.2,1.9],[1.7,1.3],[2.4,-.6],[.5,-1.7],[-1.5,-1.6]])
    fixed = np.array([idx[name] for name in selected])
    xy[fixed] = anchors
    # Reproducible display layout. This spring layout has no energetic meaning.
    for step in range(800):
        disp = xy[:, None] - xy[None, :]
        dist = np.maximum(np.linalg.norm(disp, axis=2), .08)
        force = np.sum(disp / dist[:, :, None] ** 3, axis=1) * .22
        for i, j in pairs:
            diff = xy[j] - xy[i]
            force[i] += .08 * diff; force[j] -= .08 * diff
        xy += np.clip(force, -.3, .3) * (.3 * (1 - step / 900))
        xy[fixed] = anchors
    route_edges = {frozenset(p) for p in zip(selected, selected[1:])}
    fig, ax = plt.subplots(figsize=(11, 7.2))
    fig.subplots_adjust(top=.76, bottom=.12)
    for edge in edges:
        a, b = edge["endpoints"]
        if a not in idx or b not in idx:
            continue
        selected_edge = frozenset((a, b)) in route_edges
        line = xy[[idx[a], idx[b]]]
        ax.plot(line[:, 0], line[:, 1], color=PURPLE if selected_edge else "#c7d1d7",
                lw=3 if selected_edge else 1, zorder=1 if selected_edge else 0, alpha=.9)
    aliases = {"N002": "P", "N015": "Q"}
    for name, position in zip(names, xy):
        chosen = name in selected
        ax.scatter(*position, s=390 if chosen else 65,
                   color=PURPLE if chosen else "#a8bbc5", edgecolor=PAPER, linewidth=1.5, zorder=3)
        if chosen:
            ax.text(*position, aliases.get(name, name), ha="center", va="center",
                    weight="bold", color="white", fontsize=10, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    isolated = fig.add_axes([.77,.18,.14,.20])
    for yy in (.25,.75):
        isolated.plot([.15,.7],[yy,yy],color="#c7d1d7",lw=1.5)
        isolated.scatter([.15,.7],[yy,yy],color="#a8bbc5",s=70)
    isolated.set_xlim(0,1);isolated.set_ylim(0,1);isolated.axis("off")
    isolated.set_title("4 separate minima",fontsize=9,color=MUTED)
    fig.text(.07, .94, "A route through a finite landscape", fontsize=24, weight="bold")
    fig.text(.07, .865, "36 checked minima · 32 reachable from A · 53 edge records, including reflected partners", color=MUTED)
    fig.text(.07, .05, "Purple: A → C → P → Q → D → M → B. Layout is illustrative; distance is not molecular distance or energy.\nThe minimum bottleneck in this explored graph is not a proof of the globally optimal molecular path.", color=MUTED, fontsize=10)
    save(fig, "transition_network")


if __name__ == "__main__":
    profiles(); activity(); convergence(); network()
    print("Wrote figures to", FIG)
