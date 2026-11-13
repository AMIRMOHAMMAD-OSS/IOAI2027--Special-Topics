"""Independent forward/reverse fixed-progress continuation.

A converged list of constrained minima is deliberately reported as a node
sequence. Continuity and the energy between these nodes are separate tests.
"""
from pathlib import Path
import time

import numpy as np

from .core import DATA, maxg, require, write_json
from .constraints import minimize_fixed_progress, save_case, torsions_jac, wrapped


def endpoint_targets(ctx, pattern):
    require(pattern in ("I_III", "II_IV"), "Unknown synchronized winding")
    with np.load(DATA / "collective_endpoints.npz", allow_pickle=False) as stored:
        a_xyz = stored[pattern + "_A"].ravel().copy()
        b_xyz = stored[pattern + "_B"].ravel().copy()
    require(ctx["rmsd"](ctx["points"]["A"], a_xyz) < 1e-6 and
            ctx["rmsd"](ctx["points"]["B"], b_xyz) < 1e-6,
            "Collective endpoints differ from the checked enantiomers")
    a = torsions_jac(a_xyz)[0]
    delta = wrapped(torsions_jac(b_xyz)[0] - a)
    require(np.min(np.abs(delta)) > np.deg2rad(1), "Vanishing collective torsion change")
    return a_xyz, b_xyz, a, delta


def audit_node_chords(ctx, xyz, progress, a, delta, cap=0.02):
    """Audit proposed straight connectors, including their loss of constraints.

    This does not repair a jump or label a failed connector as a valid path.
    Atom identities stay fixed. The recorded probes are finite numerical
    evidence, not a proof of a continuous supremum.
    """
    xyz = np.asarray(xyz).reshape(-1, 99, 3)
    top = -float("inf")
    maximum_constraint = 0.0
    intervals = []
    for k, (left, right) in enumerate(zip(xyz, xyz[1:])):
        raw_gap = maxg(right - left)
        pieces = max(1, int(np.ceil(raw_gap / cap)))
        peak = -float("inf")
        angular = 0.0
        for fraction in np.linspace(0, 1, 4 * pieces + 1):
            x = (1 - fraction) * left + fraction * right
            t = (1 - fraction) * progress[k] + fraction * progress[k + 1]
            peak = max(peak, ctx["eg"](x.ravel())[0])
            angular = max(angular, float(np.max(np.abs(wrapped(torsions_jac(x)[0] - a - t * delta)))))
        intervals.append({"left_index": k, "right_index": k + 1,
                          "maximum_fixed_identity_atom_gap_A": raw_gap,
                          "maximum_heavy_atom_gap_A": maxg(right[:43] - left[:43]),
                          "probe_count": 4 * pieces + 1,
                          "maximum_chord_energy_kcal_mol": peak,
                          "maximum_chord_constraint_residual_rad": angular})
        top = max(top, peak)
        maximum_constraint = max(maximum_constraint, angular)
    return {"status": "CHORDS_AUDITED_NOT_ASSUMED_FEASIBLE", "intervals": intervals,
            "maximum_sampled_chord_energy_kcal_mol": top,
            "maximum_sampled_chord_constraint_residual_rad": maximum_constraint,
            "finite_sampling_not_continuous_certificate": True,
            "fixed_atom_identities": True}


def run_branch(ctx, pattern, direction, out, intervals=40,
               gradient_tol=1e-7, constraint_tol=1e-10, audit=True):
    require(direction in ("forward", "reverse"), "Unknown continuation direction")
    require(intervals >= 4, "Use at least four progress intervals")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    a_xyz, b_xyz, a, delta = endpoint_targets(ctx, pattern)
    values = np.linspace(0.0, 1.0, intervals + 1)
    if direction == "reverse":
        values = values[::-1]
    x = a_xyz if direction == "forward" else b_xyz
    rows, coordinates, energies = [], [], []
    start = time.monotonic()
    energy0 = ctx["eg"](ctx["points"]["A"])[0]
    report = {"pattern": pattern, "direction": direction,
              "status": "RUNNING", "nodes": rows, "energy_reference_kcal_mol": energy0,
              "scope": "Independent local constrained minima; continuity is tested separately.",
              "initial_torsions_rad": a.tolist(), "torsion_delta_rad": delta.tolist(),
              "tolerances": {"constraint_rad": constraint_tol, "kkt_gradient": gradient_tol},
              "input_hashes": ctx.get("input_hashes", {}), "software": ctx["software"]}

    def checkpoint():
        report["elapsed_seconds"] = time.monotonic() - start
        if energies:
            report["maximum_node_energy_kcal_mol"] = max(energies)
            report["maximum_node_energy_above_A_kcal_mol"] = max(energies) - energy0
        write_json(out / "branch.json", report)
        np.savez_compressed(out / "nodes.npz", coordinates_angstrom=np.asarray(coordinates).reshape(-1, 99, 3),
                            progress=np.array([r["progress"] for r in rows]), energy_kcal_mol=energies)

    try:
        for k, t in enumerate(values):
            target = a + t * delta
            seed = x.copy()
            if k:
                theta, jac = torsions_jac(x)
                q = ctx["basis"](x)
                seed += q @ np.linalg.lstsq(jac @ q, -wrapped(theta - target), rcond=None)[0]
            result, trace = minimize_fixed_progress(ctx, seed, target,
                {"constraint_tol": constraint_tol, "gradient_tol": gradient_tol})
            result.update(pattern=pattern, progress=float(t), branch_direction=direction)
            save_case(out / f"node_{k:03d}", result, trace)
            candidate = np.asarray(result["xyz_A"]).ravel()
            row = {key: result[key] for key in ("status", "energy_kcal_mol", "max_constraint_rad",
                "kkt_gradient_max_atomic_norm", "projected_gradient_max_atomic_norm",
                "lowest_tangent_curvature", "negative_tangent_modes", "elapsed_seconds")}
            row.update(progress=float(t), directory=f"node_{k:03d}",
                       maximum_fixed_identity_atom_step_A=maxg(candidate - x),
                       slsqp_iterations=result["slsqp"]["iterations"],
                       profile_derivative_kcal_mol_per_unit_progress=-float(np.dot(
                           result["multipliers_kcal_mol_per_rad"], delta)))
            rows.append(row)
            coordinates.append(candidate.reshape(99, 3))
            energies.append(result["energy_kcal_mol"])
            checkpoint()
            if k % 10 == 0 or k == intervals or result["status"] != "CONSTRAINED_MINIMUM_VERIFIED":
                print(f"{pattern} {direction} {k}/{intervals}: E-A={energies[-1]-energy0:.6f}; "
                      f"KKT={row['kkt_gradient_max_atomic_norm']:.2g}; {row['status']}", flush=True)
            require(result["status"] == "CONSTRAINED_MINIMUM_VERIFIED",
                    "Constrained node requires review; partial results were saved")
            x = candidate
        report["endpoint_symmetry_rmsd_to_A_B"] = {
            label: ctx["rmsd"](ctx["points"][label], x) for label in ("A", "B")}
        expected = "B" if direction == "forward" else "A"
        require(report["endpoint_symmetry_rmsd_to_A_B"][expected] < 0.002,
                "Constrained branch ended in a different basin")
        report["status"] = "COMPLETE_CONSTRAINED_NODE_SEQUENCE"
        if audit:
            chord_report = audit_node_chords(ctx, coordinates, values, a, delta)
            write_json(out / "chord_audit.json", chord_report)
            report["chord_audit"] = "chord_audit.json"
        checkpoint()
        return report
    except Exception as error:
        report.update(status="PARTIAL_OR_UNRESOLVED_NODE_SEQUENCE", error=str(error))
        checkpoint()
        raise
