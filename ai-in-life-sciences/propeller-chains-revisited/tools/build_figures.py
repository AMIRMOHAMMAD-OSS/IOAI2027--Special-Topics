#!/usr/bin/env python3
"""Regenerate original figures from the checked, machine-readable reports."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REF = ROOT / "results" / "reference"
FIG.mkdir(exist_ok=True)
INK, MUTED, PAPER = "#25364a", "#52677d", "#ffffff"
DARK, MID, LIGHT = [matplotlib.colormaps["Blues"](v) for v in (.88, .70, .48)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#bdd1e5",
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
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    fig.subplots_adjust(top=.96, bottom=.17, left=.075, right=.985, wspace=.12)
    x, y = np.array(p["normalized_arc_length"]), np.array(p["energy_above_A_kcal_mol"])
    axes[0].plot(x, y, color=DARK, lw=2.0, label=f"Localized · {y.max():.3f} kcal/mol")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    aliases = {"N002": "P", "N015": "Q"}
    for label, point in p["markers"].items():
        if not label.startswith(("TS", "C_pair", "rescue", "R_N")):
            axes[0].scatter(point["progress"], point["energy_above_A_kcal_mol"], s=29, color=DARK, zorder=3)
            axes[0].annotate(aliases.get(label, label), (point["progress"], point["energy_above_A_kcal_mol"]),
                             xytext=(0, -18), textcoords="offset points", ha="center", color=DARK)
    axes[0].set_xlabel("Normalized path length")
    axes[0].set_ylabel("Energy above A [kcal/mol]")
    collective = REF / "collective_profiles.json"
    if collective.exists():
        series = json.loads(collective.read_text())
        for i, branch in enumerate(series["branches"]):
            color = MID if branch["pattern"] == "I_III" else LIGHT
            t = np.array(branch["progress"])
            energy = np.array(branch["energy_above_A_kcal_mol"])
            marker = "o" if branch["direction"] == "forward" else "+"
            axes[1].scatter(t, energy, s=10, marker=marker, color=color, alpha=.20)
        for pattern, color, style in (("I_III", MID, "-"), ("II_IV", LIGHT, "--")):
            with np.load(REF / (pattern + "_complete_display.npz")) as z:
                axes[1].plot(z["progress"], z["energy_above_A_kcal_mol"], color=color, lw=2.0, ls=style,
                    label=pattern.replace("_", "/") + f"  {z['energy_above_A_kcal_mol'].max():.3f} kcal/mol")
        axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_xlabel("Imposed rotation, t")
    for ax in axes:
        ax.set_ylim(-1.2, 18)
        ax.set_xlim(-.025, 1.025)
        ax.axhline(7.40399469869, color=DARK, ls="--", lw=1, alpha=.7)
        ax.grid(axis="y", color="#deebf7", lw=.7)
    save(fig, "energy_profiles")


def activity():
    d = json.loads((REF / "mechanism_geometry.json").read_text())
    rows = d["events"]
    data = 100 * np.array([r["pair_squared_torsion_derivative_fractions"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    fig.subplots_adjust(top=.97, bottom=.17, left=.16, right=.88)
    image = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(6):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center",
                    color="white" if data[i,j] > 60 else INK, fontsize=11)
    ax.set_xticks(range(3), ["I–II", "II–III", "III–IV"])
    ax.set_yticks(range(6), ["A → C", "C → P", "P → Q", "Q → D", "D → M", "M → B"])
    ax.set_xlabel("Adjacent ring pair", labelpad=10)
    fig.colorbar(image, ax=ax, fraction=.045, pad=.05, label="Relative torsion contribution [%]")
    save(fig, "torsion_activity")


def convergence():
    path = REF / "collective_profiles.json"
    if not path.exists():
        return
    profiles = json.loads(path.read_text())["branches"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    fig.subplots_adjust(top=.96, bottom=.27, left=.075, right=.985, wspace=.39)
    fields = ["max_constraint_rad", "kkt_gradient_max_atomic_norm", "lowest_tangent_curvature"]
    labels = ["Angle residual [rad]", "Residual force [kcal/(mol Å)]", "Minimum constrained curvature"]
    for branch in profiles:
        color = MID if branch["pattern"] == "I_III" else LIGHT
        marker = "o" if branch["direction"] == "forward" else "+"
        for ax, key in zip(axes, fields):
            y = np.asarray(branch[key])
            if key != "lowest_tangent_curvature":
                y = np.maximum(y, 1e-16)
            ax.scatter(branch["progress"], y, s=15, marker=marker, color=color, alpha=.7)
    for ax, label in zip(axes, labels):
        ax.set_xlabel("Imposed rotation, t")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(color="#deebf7", lw=.7)
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[0].axhline(1e-10, color=INK, ls="--", lw=.8)
    axes[1].axhline(1e-7, color=INK, ls="--", lw=.8)
    axes[2].axhline(0, color=INK, lw=.8)
    handles = [Line2D([], [], color=MID, lw=2, label="I/III"),
               Line2D([], [], color=LIGHT, lw=2, label="II/IV"),
               Line2D([], [], color=DARK, marker="o", ls="none", ms=4, label="Forward"),
               Line2D([], [], color=DARK, marker="+", ls="none", ms=6, label="Reverse")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
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
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.subplots_adjust(top=.99, bottom=.07, left=.02, right=.98)
    for edge in edges:
        a, b = edge["endpoints"]
        if a not in idx or b not in idx:
            continue
        selected_edge = frozenset((a, b)) in route_edges
        line = xy[[idx[a], idx[b]]]
        ax.plot(line[:, 0], line[:, 1], color=DARK if selected_edge else "#c6dbef",
                lw=3 if selected_edge else 1, zorder=1 if selected_edge else 0, alpha=.9)
    aliases = {"N002": "P", "N015": "Q"}
    for name, position in zip(names, xy):
        chosen = name in selected
        ax.scatter(*position, s=390 if chosen else 65,
                   color=DARK if chosen else "#9ecae1", edgecolor=PAPER, linewidth=1.5, zorder=3)
        if chosen:
            ax.text(*position, aliases.get(name, name), ha="center", va="center",
                    color="white", fontsize=10, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    isolated = fig.add_axes([.81,.16,.13,.19])
    for yy in (.25,.75):
        isolated.plot([.15,.7],[yy,yy],color="#c6dbef",lw=1.5)
        isolated.scatter([.15,.7],[yy,yy],color="#9ecae1",s=70)
    isolated.set_xlim(0,1);isolated.set_ylim(0,1);isolated.axis("off")
    fig.legend(handles=[Line2D([], [], color=DARK, marker="o", lw=2, label="Selected path"),
                        Line2D([], [], color="#9ecae1", marker="o", lw=1, label="Other connections")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, "transition_network")


if __name__ == "__main__":
    profiles(); activity(); convergence(); network()
    print("Wrote figures to", FIG)
