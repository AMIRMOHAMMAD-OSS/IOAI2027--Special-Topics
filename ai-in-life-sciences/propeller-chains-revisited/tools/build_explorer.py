#!/usr/bin/env python3
"""Build an offline, dependency-free molecular viewer from recorded coordinates."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/reference"
model = json.loads((ROOT / "src/propeller_paths/data/molecule.json").read_text())
DATA = {"bonds": [[a,b,order] for a,b,order in model["bonds"] if a < 43 and b < 43], "series": {}}


def add_series(key, title, filename, kind, note):
    with np.load(REF / filename, allow_pickle=False) as z:
        xyz = z["coordinates_angstrom"].copy()
        # One translation and one proper display rotation per entire sequence.
        # This is a camera operation, not atom-by-atom fitting or a data edit.
        center = xyz[0].mean(0)
        xyz -= center
        DATA["series"][key] = {"title": title, "kind": kind, "note": note,
            "axis": "arc" if key == "sequential" else "collective",
            "xyz": np.round(xyz, 5).tolist(), "energy": np.round(z["energy_above_A_kcal_mol"], 7).tolist(),
            "progress": np.round(z["progress"], 7).tolist()}


add_series("sequential", "Localized A → C → P → Q → D → M → B", "sequential_display.npz", "path",
    "Recorded configurations from a validated six-saddle path. Five adjacent-ring disrotations and one methyl/conformational adjustment.")
for pattern in ("I_III", "II_IV"):
    add_series(pattern + "_complete", "Synchronized " + pattern.replace("_", "/") + " · checked complete candidate",
        pattern + "_complete_display.npz", "path",
        "Complete candidate with converged branch nodes, explicit fixed-progress saddle connectors, and checked interpolation. Exact constraints at recorded nodes; tiny measured between-frame angular residual. This is not a globally minimized collective barrier.")
for pattern in ("I_III", "II_IV"):
    for direction in ("forward", "reverse"):
        add_series(pattern+"_"+direction, pattern.replace("_","/")+" · "+direction+" minima",
            pattern+"_"+direction+"_nodes_display.npz", "nodes",
            "Converged fixed-progress minima. Browse individual points: gaps between minima require separate physical connectors and can conceal higher energies.")

template = (ROOT / "tools/explorer_template.html").read_text()
html = template.replace("__PROPELLER_DATA__", json.dumps(DATA, separators=(",",":"), allow_nan=False))
(ROOT / "docs/explorer.html").write_text(html)
print("Wrote", ROOT / "docs/explorer.html", "bytes", len(html.encode()))
