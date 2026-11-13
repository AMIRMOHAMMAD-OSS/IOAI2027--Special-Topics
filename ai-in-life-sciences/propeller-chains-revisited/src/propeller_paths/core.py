"""Explicit molecular model, derivatives, stationary points, and proper matching.

Coordinates are angstrom; bare MMFF94 energies are kcal/mol. The force field is
mutable and serial: do not share one context between concurrent threads.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment, root
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors

DATA = Path(__file__).parent / "data"
STATES = ("A", "C", "P", "Q", "D", "M", "B")
EDGES = (("A", "TS_AC", "C"), ("C", "TS_CP", "P"),
         ("P", "TS_PQ", "Q"), ("Q", "TS_QD", "D"),
         ("D", "TS_DM", "M"), ("M", "TS_MB", "B"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    """Replace one report atomically; callers create a new run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_default,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software_versions():
    return {"python": sys.version, "numpy": np.__version__,
            "scipy": scipy.__version__, "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform()}


def maxg(vector):
    """Maximum Euclidean atomic-vector norm, not componentwise maximum."""
    return float(np.linalg.norm(np.asarray(vector).reshape(-1, 3), axis=1).max())


def basis(x):
    """Orthonormal Cartesian complement to three translations/three rotations."""
    xyz = np.asarray(x, dtype=float).reshape(-1, 3)
    centered = xyz - xyz.mean(0)
    rigid = np.column_stack([np.tile(a, len(xyz)) for a in np.eye(3)] +
                            [np.cross(a, centered).ravel() for a in np.eye(3)])
    require(np.linalg.matrix_rank(rigid) == 6, "Rigid-mode rank must be six")
    return np.linalg.qr(rigid, mode="complete")[0][:, 6:]


def create_context(graph, records):
    """Build from plain explicit-atom data; no previous scripts are executed."""
    editable = Chem.RWMol()
    for number in graph["atoms"]:
        atom = Chem.Atom(int(number))
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    for a, b, order in graph["bonds"]:
        require(order in (1.0, 1.5), "Unsupported graph bond order")
        editable.AddBond(int(a), int(b), Chem.BondType.AROMATIC
                         if order == 1.5 else Chem.BondType.SINGLE)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    require(molecule.GetNumAtoms() == 99 and
            rdMolDescriptors.CalcMolFormula(molecule) == "C43H56",
            "Expected the explicit 99-atom C43H56 model")
    initial = np.asarray(records["A"]["xyz_A"], dtype=float).reshape(99, 3)
    conformer = Chem.Conformer(99)
    conformer.Set3D(True)
    for i, position in enumerate(initial):
        conformer.SetAtomPosition(i, tuple(map(float, position)))
    molecule.AddConformer(conformer)
    properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94")
    require(properties is not None, "MMFF94 parameters unavailable")
    field = AllChem.MMFFGetMoleculeForceField(molecule, properties,
                                             nonBondedThresh=100.0)
    field.Initialize()

    def eg(x):
        xyz = np.asarray(x, dtype=float).reshape(297)
        require(np.isfinite(xyz).all(), "Nonfinite Cartesian coordinates")
        positions = tuple(xyz)
        # RDKit's energy call refreshes the position-dependent internal cache.
        energy = float(field.CalcEnergy(positions))
        gradient = np.asarray(field.CalcGrad(positions), dtype=float)
        require(np.isfinite(energy) and np.isfinite(gradient).all(),
                "Nonfinite force-field result")
        return energy, gradient

    def hessian(x, step=1e-4):
        x = np.asarray(x, dtype=float).reshape(297)
        matrix = np.empty((297, 297))
        for j in range(297):
            delta = np.zeros(297)
            delta[j] = step
            matrix[:, j] = (eg(x + delta)[1] - eg(x - delta)[1]) / (2 * step)
        asymmetry = np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30)
        require(asymmetry < 1e-5, "Numerical Hessian asymmetry exceeds tolerance")
        return (matrix + matrix.T) / 2

    def refine(x, expected_index, displacement_cap=0.1):
        initial = np.asarray(x, dtype=float).reshape(297)
        q = basis(initial)

        def coordinates(y):
            value = initial + q @ y
            require(maxg(value - initial) < displacement_cap,
                    "Stationary refinement exceeded displacement cap")
            return value

        result = root(lambda y: q.T @ eg(coordinates(y))[1], np.zeros(291),
                      jac=lambda y: q.T @ hessian(coordinates(y)) @ q,
                      method="hybr", options={"xtol": 1e-9, "maxfev": 100, "factor": 0.1})
        x = coordinates(result.x)
        energy, gradient = eg(x)
        q = basis(x)
        values, vectors = np.linalg.eigh(q.T @ hessian(x) @ q)
        index = int(np.sum(values < -1e-4))
        require(maxg(gradient) < 1e-4, "Residual stationary-point force")
        require(index == expected_index and np.min(np.abs(values)) > 1e-4,
                "Wrong or unresolved stationary-point index")
        record = {"energy_kcal_mol": energy, "max_gradient_kcal_mol_A": maxg(gradient),
                  "negative_internal_modes": index, "internal_modes": 291,
                  "lowest_curvatures": values[:5].tolist(),
                  "solver_success": bool(result.success), "solver_message": str(result.message),
                  "xyz_A": x.reshape(99, 3).tolist()}
        return x, q @ vectors[:, 0], record

    heavy = np.array([a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() > 1])
    heavy_molecule = Chem.RemoveHs(molecule)
    mappings = heavy_molecule.GetSubstructMatches(heavy_molecule, uniquify=False,
                                                 useChirality=False, maxMatches=100000)
    require(0 < len(mappings) < 100000, "Incomplete graph automorphism enumeration")
    hydrogens = {i: [n.GetIdx() for n in molecule.GetAtomWithIdx(int(i)).GetNeighbors()
                     if n.GetAtomicNum() == 1] for i in heavy}
    ctx = {"eg": eg, "maxg": maxg, "basis": basis, "hessian": hessian, "refine": refine,
           "molecule": molecule, "properties": properties, "heavy": heavy,
           "maps": mappings, "hydrogens": hydrogens, "data": graph, "records": records,
           "points": {k: np.asarray(v["xyz_A"], dtype=float).reshape(297)
                      for k, v in records.items()},
           "software": software_versions(), "automorphism_count": len(mappings)}
    ctx["fit_transform"] = lambda reference, candidate: fit_transform(ctx, reference, candidate)
    ctx["rmsd"] = lambda reference, candidate: fit_transform(ctx, reference, candidate)["all_atom_rmsd_A"]
    return ctx


def load_context(data_directory=None):
    folder = DATA if data_directory is None else Path(data_directory)
    graph_path = folder / "molecule.json"
    points_path = folder / "stationary_points.json"
    ctx = create_context(json.loads(graph_path.read_text()),
                         json.loads(points_path.read_text()))
    ctx["input_hashes"] = {"molecule.json": sha256(graph_path),
                           "stationary_points.json": sha256(points_path)}
    return ctx


def fit_transform(ctx, reference, candidate):
    """Graph-preserving permutation plus proper rotation; never a reflection.

    This operation identifies basins. For path assembly apply a chosen mapping
    to a whole segment, never choose a new permutation independently per frame.
    """
    a = np.asarray(reference).reshape(99, 3)
    b = np.asarray(candidate).reshape(99, 3)
    heavy, hydrogens = ctx["heavy"], ctx["hydrogens"]
    ca, cb = a[heavy].mean(0), b[heavy].mean(0)
    best = None
    for mapping in ctx["maps"]:
        perm = np.arange(99)
        perm[heavy] = heavy[np.asarray(mapping)]
        u, _, vt = np.linalg.svd((b[perm[heavy]] - cb).T @ (a[heavy] - ca))
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        shifted = (b - cb) @ rotation + ca
        for i, j in zip(heavy, perm[heavy]):
            hi, hj = hydrogens[i], hydrogens[j]
            if hi:
                rows, cols = linear_sum_assignment(np.sum(
                    (a[np.array(hi), None] - shifted[np.array(hj)][None]) ** 2, axis=2))
                perm[np.array(hi)[rows]] = np.array(hj)[cols]
        norms = np.linalg.norm(shifted[perm] - a, axis=1)
        score = float(np.sqrt(np.mean(norms ** 2)))
        if best is None or score < best["all_atom_rmsd_A"]:
            best = {"permutation": perm.tolist(), "rotation": rotation.tolist(),
                    "translation": (ca - cb @ rotation).tolist(),
                    "determinant": float(np.linalg.det(rotation)),
                    "all_atom_rmsd_A": score,
                    "heavy_rmsd_A": float(np.sqrt(np.mean(norms[heavy] ** 2))),
                    "max_atom_residual_A": float(norms.max())}
    perm, molecule = best["permutation"], ctx["molecule"]
    require(sorted(perm) == list(range(99)) and best["determinant"] > 0.99999999,
            "Invalid proper atom mapping")
    for atom in molecule.GetAtoms():
        other = molecule.GetAtomWithIdx(perm[atom.GetIdx()])
        require((atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge()) ==
                (other.GetAtomicNum(), other.GetIsotope(), other.GetFormalCharge()),
                "Atom mapping changes atom type")
    for bond in molecule.GetBonds():
        other = molecule.GetBondBetweenAtoms(perm[bond.GetBeginAtomIdx()], perm[bond.GetEndAtomIdx()])
        require(other is not None and other.GetBondType() == bond.GetBondType(),
                "Atom mapping changes molecular connectivity")
    best["graph_automorphism_verified"] = True
    return best


def verify_core(ctx, stationary=True):
    """Independent cache, derivative, enantiomer, and saved-point checks."""
    eg, points = ctx["eg"], ctx["points"]
    probe = points["C"] + np.random.default_rng(20261112).normal(0, 0.001, 297)
    before = eg(probe)
    eg(points["B"])
    after = eg(probe)
    cache = {"energy_error": abs(before[0] - after[0]),
             "gradient_error": float(np.max(np.abs(before[1] - after[1])))}
    require(cache["energy_error"] < 1e-9 and cache["gradient_error"] < 1e-7,
            "Force-field cache round trip failed")
    errors = []
    for seed in (1, 20261113):
        direction = np.random.default_rng(seed).normal(size=297)
        direction /= np.linalg.norm(direction)
        analytic = float(eg(probe)[1] @ direction)
        numeric = (eg(probe + 1e-5 * direction)[0] - eg(probe - 1e-5 * direction)[0]) / 2e-5
        errors.append(abs(analytic - numeric))
    require(max(errors) < 1e-4, "MMFF94 directional derivative check failed")
    a, b = points["A"], points["B"]
    mirror = a.reshape(99, 3) * [-1.0, 1.0, 1.0]
    proper, reflected = ctx["rmsd"](a, b), ctx["rmsd"](mirror, b)
    require(proper > 0.02 and reflected < 0.002, "Enantiomer identity check failed")
    report = {"cache_roundtrip": cache, "directional_gradient_errors": errors,
              "proper_A_B_rmsd_A": proper, "reflected_A_B_rmsd_A": reflected,
              "reflection_only_used_for_identity_diagnostic": True,
              "automorphism_count": len(ctx["maps"]), "stationary_points": {}}
    if stationary:
        for label, x in points.items():
            e, g = eg(x)
            q = basis(x)
            values = np.linalg.eigvalsh(q.T @ ctx["hessian"](x) @ q)
            expected = int(label.startswith("TS_"))
            energy_error = e - ctx["records"][label]["energy_kcal_mol"]
            require(abs(energy_error) < 0.001 and maxg(g) < 1e-4,
                    label + " force-field check failed")
            require(int(np.sum(values < -1e-4)) == expected and np.min(np.abs(values)) > 1e-4,
                    label + " stationary index failed")
            report["stationary_points"][label] = {
                "energy_kcal_mol": e, "reference_energy_error": energy_error,
                "max_gradient_kcal_mol_A": maxg(g), "negative_internal_modes": expected,
                "lowest_five_curvatures": values[:5].tolist()}
    report["status"] = "MODEL_AND_REFERENCE_CHECKS_PASSED"
    return report
