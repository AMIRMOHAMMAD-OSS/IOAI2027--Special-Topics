"""Revalidate the selected localized route with explicit, fixed-atom paths.

This module imports the packaged model through :mod:`propeller_paths.core`.
It never reads or executes an earlier chat script. Coordinates are angstrom;
energies and Cartesian derivatives use kcal/mol and angstrom. The paths are
numerically checked downhill/relaxation paths, not mass-weighted IRCs.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .core import EDGES, STATES, require, sha256, write_json

TORSIONS = np.array([[2, 3, 24, 6], [3, 24, 6, 7], [8, 9, 25, 12],
                     [9, 25, 12, 13], [14, 15, 26, 18], [15, 26, 18, 19]])
GRADIENT_TOLERANCE = 1e-5
PATH_ENERGY_TOLERANCE = 1e-5
ATOM_STEP_CAP = 0.02
ENDPOINT_TOLERANCE = 0.002


def angles(x):
    """Six signed ring/reference-plane torsions in the immutable atom order."""
    p = np.asarray(x).reshape(99, 3)[TORSIONS]
    a, b, c = p[:, 0] - p[:, 1], p[:, 2] - p[:, 1], p[:, 3] - p[:, 2]
    length = np.linalg.norm(b, axis=1)
    require(np.min(length) > 1e-8, "Singular torsion bond")
    b = b / length[:, None]
    v = a - np.sum(a * b, axis=1)[:, None] * b
    w = c - np.sum(c * b, axis=1)[:, None] * b
    require(np.min(np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)) > 1e-10,
            "Singular torsion reference plane")
    return np.degrees(np.arctan2(np.sum(np.cross(b, v) * w, axis=1),
                                np.sum(v * w, axis=1)))


def wrapped_degrees(value):
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def capped_minimize(eg, initial, capture, cap=ATOM_STEP_CAP, iterations=4000):
    """Armijo L-BFGS, recording every accepted step with an atomic step cap.

    A stopped minimizer is still only a proposal. The caller must refine and
    check its stationary index and audit the final refinement connector.
    """
    require(cap > 0 and iterations > 0, "Invalid minimization budget")
    x = np.asarray(initial, dtype=float).reshape(297).copy()
    energy, gradient = eg(x)
    memory, status, accepted = [], "ITERATION_LIMIT", 0
    for iteration in range(iterations):
        if np.linalg.norm(gradient.reshape(99, 3), axis=1).max() < GRADIENT_TOLERANCE:
            status = "GRADIENT_CONVERGED"
            break
        q, coefficients = gradient.copy(), []
        for s, y, rho in reversed(memory):
            coefficient = rho * float(s @ q)
            coefficients.append(coefficient)
            q -= coefficient * y
        gamma = (np.clip(float(memory[-1][0] @ memory[-1][1]) /
                         float(memory[-1][1] @ memory[-1][1]), 1e-7, 0.1)
                 if memory else 0.001)
        direction = gamma * q
        for (s, y, rho), coefficient in zip(memory, reversed(coefficients)):
            direction += s * (coefficient - rho * float(y @ direction))
        direction = -direction
        if float(gradient @ direction) >= -1e-16:
            memory, direction = [], -0.001 * gradient
        largest = float(np.linalg.norm(direction.reshape(99, 3), axis=1).max())
        if largest > cap:
            direction *= cap / largest
        for backtrack in range(25):
            step = direction * 0.5 ** backtrack
            trial = x + step
            trial_energy, trial_gradient = eg(trial)
            if trial_energy <= energy + 1e-4 * float(gradient @ step):
                break
        else:
            status = "LINESEARCH_STOP"
            break
        y = trial_gradient - gradient
        sy = float(step @ y)
        if sy > 1e-12 * np.linalg.norm(step) * np.linalg.norm(y):
            memory.append((step.copy(), y.copy(), 1.0 / sy))
            memory = memory[-20:]
        x, energy, gradient = trial, trial_energy, trial_gradient
        capture(x)
        accepted += 1
    return x, {"status": status, "iterations": iteration + 1,
               "accepted_steps": accepted, "cap_angstrom": cap,
               "final_gradient_max_atomic_norm": float(np.linalg.norm(
                   gradient.reshape(99, 3), axis=1).max())}


def audit_trace(ctx, positions, top, subdivision_cap=ATOM_STEP_CAP,
                anchor_indices=()):
    """Audit fixed-atom chords and their quarter points against an energy cap.

    Subdivision improves the sampling of the supplied path; it does not repair
    its geometry or any energy overshoot. Every original image is retained.
    Optional input-image anchors are mapped to exact output-image indices.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 99, 3)
    require(len(positions) > 0 and np.isfinite(positions).all(), "Invalid path")
    require(subdivision_cap > 0 and np.isfinite(top), "Invalid path-check limit")
    dense, original_to_dense, raw_gap = [positions[0].copy()], [0], 0.0
    for left, right in zip(positions, positions[1:]):
        gap = float(np.linalg.norm(right - left, axis=1).max())
        raw_gap = max(raw_gap, gap)
        # Tolerance prevents splitting an already capped interval solely from
        # roundoff. The final measured gap is checked independently below.
        pieces = max(1, int(np.ceil(gap / subdivision_cap - 1e-12)))
        dense.extend(left + (right - left) * i / pieces for i in range(1, pieces))
        dense.append(right.copy())
        original_to_dense.append(len(dense) - 1)
    dense = np.asarray(dense)
    energies = np.asarray([ctx["eg"](x.ravel())[0] for x in dense])
    require(np.isfinite(energies).all(), "Nonfinite path energy")
    maximum_index = int(np.argmax(energies))
    peak = float(energies[maximum_index])
    peak_location = {"kind": "stored_image", "image_index": maximum_index}
    max_gap, probe_count = 0.0, 0
    for index, (left, right) in enumerate(zip(dense, dense[1:])):
        max_gap = max(max_gap, float(np.linalg.norm(right - left, axis=1).max()))
        for fraction in (0.25, 0.5, 0.75):
            energy = float(ctx["eg"](((1 - fraction) * left + fraction * right).ravel())[0])
            require(np.isfinite(energy), "Nonfinite gap-probe energy")
            probe_count += 1
            if energy > peak:
                peak = energy
                peak_location = {"kind": "interval_probe", "left_image_index": index,
                                 "fraction": fraction}
    passed = (peak <= top + PATH_ENERGY_TOLERANCE and
              max_gap <= subdivision_cap * (1 + 1e-10))
    anchors = {}
    for index in anchor_indices:
        require(0 <= index < len(positions), "Path marker outside input trace")
        anchors[str(index)] = original_to_dense[index]
    return dense, energies, {
        "status": "VERIFIED_DISCRETE_PATH" if passed else "CONTINUITY_UNRESOLVED",
        "fixed_atom_identities": True, "dynamic_per_frame_remapping": False,
        "original_images": len(positions), "stored_images": len(dense),
        "interval_probe_count": probe_count,
        "maximum_original_interval_atom_displacement_A": raw_gap,
        "maximum_checked_interval_atom_displacement_A": max_gap,
        "maximum_stored_energy_kcal_mol": float(energies.max()),
        "maximum_sampled_energy_kcal_mol": peak, "maximum_location": peak_location,
        "energy_cap_kcal_mol": float(top),
        "sampled_excess_above_saddle_kcal_mol": max(0.0, peak - top),
        "energy_tolerance_kcal_mol": PATH_ENERGY_TOLERANCE,
        "subdivision_cap_A": subdivision_cap, "probe_fractions": [0.25, 0.5, 0.75],
        "anchor_image_indices": anchors,
        "finite_sampling_not_continuous_certificate": True}


def join_to_node(ctx, endpoint, reference):
    """Map one stored node into the branch endpoint frame with a proper fit."""
    endpoint = np.asarray(endpoint).reshape(99, 3)
    reference = np.asarray(reference).reshape(99, 3)
    fit = ctx["fit_transform"](endpoint, reference)
    aligned = (reference[fit["permutation"]] @ np.asarray(fit["rotation"]) +
               np.asarray(fit["translation"]))
    gap = float(np.linalg.norm(aligned - endpoint, axis=1).max())
    require(fit["all_atom_rmsd_A"] < ENDPOINT_TOLERANCE and gap < ENDPOINT_TOLERANCE,
            "Stationary endpoint-to-node connector is too large")
    return aligned, {"fit_transform": fit, "maximum_atom_displacement_A": gap,
                     "direction": "stored_node_into_branch_endpoint_frame",
                     "fixed_mapping": True}


def downhill(ctx, saddle, mode, sign):
    """Record mode displacement, descent, capped relaxation, and final root."""
    require(sign in (-1, 1), "Downhill sign must be -1 or +1")
    eg, maxg = ctx["eg"], ctx["maxg"]
    saddle = np.asarray(saddle).reshape(297)
    top = eg(saddle)[0]
    require(maxg(mode) > 0, "Zero unstable mode")
    mode = np.asarray(mode).reshape(297) / maxg(mode)
    for reduction in range(8):
        displacement = sign * 0.03 * 0.5 ** reduction * mode
        x = saddle + displacement
        if eg(x)[0] < top - 1e-10:
            break
    else:
        raise RuntimeError("No downhill initial mode displacement")
    descent, descent_energy, minimization, minimization_energy = [], [], [], []
    for iteration in range(251):
        energy, gradient = eg(x)
        descent.append(x.reshape(99, 3).copy())
        descent_energy.append(energy)
        gradient_max = maxg(gradient)
        if iteration == 250 or (top - energy > 0.05 and gradient_max < 0.02):
            break
        require(gradient_max > 1e-14, "Descent stalled before basin relaxation")
        step = -0.008 * gradient / gradient_max
        for backtrack in range(20):
            alpha = 0.5 ** backtrack
            trial = x + alpha * step
            if eg(trial)[0] <= energy + 1e-4 * alpha * float(gradient @ step):
                x = trial
                break
        else:
            raise RuntimeError("Downhill line search stalled")

    def capture(value):
        minimization.append(np.asarray(value).reshape(99, 3).copy())
        minimization_energy.append(eg(value)[0])

    capture(x)
    polished, optimization = capped_minimize(eg, x, capture)
    endpoint, _, record = ctx["refine"](polished, 0)
    require(record["max_gradient_kcal_mol_A"] < GRADIENT_TOLERANCE,
            "Downhill endpoint has excessive residual force")
    raw = np.asarray([saddle.reshape(99, 3)] + descent + minimization +
                     [polished.reshape(99, 3), endpoint.reshape(99, 3)])
    raw_energy = np.asarray([eg(value.ravel())[0] for value in raw])
    increase = max(0.0, float(np.max(np.diff(raw_energy))))
    require(increase <= 1e-7 and record["energy_kcal_mol"] <= top + 1e-7,
            "Recorded downhill energies failed the monotonicity check")
    record.update(stationary=True, descent_steps=len(descent) - 1,
                  capped_minimization=optimization,
                  maximum_recorded_energy_increase_kcal_mol=increase,
                  initial_mode_displacement_max_atom_A=maxg(displacement),
                  final_root_connector_max_atom_A=maxg(endpoint - polished))
    arrays = {"descent_xyz_A": np.asarray(descent),
              "descent_energy_kcal_mol": np.asarray(descent_energy),
              "minimization_xyz_A": np.asarray(minimization),
              "minimization_energy_kcal_mol": np.asarray(minimization_energy),
              "polished_xyz_A": polished.reshape(99, 3),
              "endpoint_xyz_A": endpoint.reshape(99, 3),
              "raw_full_xyz_A": raw, "raw_full_energy_kcal_mol": raw_energy}
    return endpoint, record, arrays


def _mode_description(ctx, x, mode):
    scaled = 0.02 * mode / ctx["maxg"](mode)
    derivative = wrapped_degrees(angles(x + scaled) - angles(x - scaled)) / 0.04
    power = np.sum(derivative.reshape(3, 2) ** 2, axis=1)
    total = float(power.sum())
    return {"derivative_deg_per_max_atom_A": derivative.tolist(),
            "dominant_pair": int(np.argmax(power)) + 1,
            "sum_squared_six_torsion_derivatives": total,
            "fraction_of_six_torsion_squared_derivatives":
                float(power.max() / total) if total > 1e-12 else None,
            "scope": "Geometric descriptor in this saddle's atom frame; not a kinetic-energy fraction or global ring numbering."}


def _check_edge(ctx, out, left, label, right, mode):
    saddle, record = ctx["points"][label], ctx["records"][label]
    top = record["energy_kcal_mol"]
    edge = {"id": label, "left": left, "right": right, "saddle": record,
            "branches": {}, "status": "BRANCH_VALIDATION_RUNNING"}
    paths = {}
    write_json(out / (label + ".json"), edge)
    for sign in (-1, 1):
        endpoint, minimum, arrays = downhill(ctx, saddle, mode, sign)
        # Persist raw evidence before matching or gap checks can reject it.
        raw_name = label + ("_minus_raw.npz" if sign == -1 else "_plus_raw.npz")
        np.savez_compressed(out / raw_name, **arrays)
        matches = [name for name in STATES
                   if abs(ctx["records"][name]["energy_kcal_mol"] - minimum["energy_kcal_mol"]) < 0.001
                   and ctx["rmsd"](ctx["points"][name], endpoint) < ENDPOINT_TOLERANCE]
        require(len(matches) == 1 and matches[0] in (left, right),
                label + " reached an unexpected or ambiguous endpoint: " + repr(matches))
        name = matches[0]
        require(name not in paths, label + " returns to the same basin on both sides")
        aligned, join = join_to_node(ctx, endpoint, ctx["points"][name])
        joined = np.concatenate([arrays["raw_full_xyz_A"], aligned[None]], axis=0)
        dense, energies, audit = audit_trace(ctx, joined, top)
        branch_name = label + "_" + name + ".npz"
        np.savez_compressed(out / branch_name, coordinates_angstrom=dense,
                            energy_kcal_mol=energies, fixed_atom_ids=np.arange(99))
        edge["branches"][name] = {"sign": sign, "minimum": minimum,
                                   "path_check": audit, "node_join": join,
                                   "raw_trace_file": raw_name, "path_file": branch_name,
                                   "raw_trace_sha256": sha256(out / raw_name),
                                   "path_sha256": sha256(out / branch_name)}
        write_json(out / (label + ".json"), edge)
        require(audit["status"] == "VERIFIED_DISCRETE_PATH",
                label + " has an unresolved path interval")
        paths[name] = dense
    require(set(paths) == {left, right}, label + " did not connect the requested basins")
    edge.update(status="INDEX_ONE_CONNECTION_REVALIDATED",
                angular_mode=_mode_description(ctx, saddle, mode),
                maximum_sampled_energy_kcal_mol=max(
                    top, *(b["path_check"]["maximum_sampled_energy_kcal_mol"]
                           for b in edge["branches"].values())))
    write_json(out / (label + ".json"), edge)
    print(f"{label}: E={top:.8f}; index=1; connects {left} and {right}", flush=True)
    return edge, paths


def run_sequential(ctx, out):
    """Refine 13 packaged points and revalidate the six-edge selected route.

    ``out`` must not exist. The mutable serial context receives the refined
    points in memory; shipped input data remain unchanged. Partial records are
    retained if a check fails or execution is interrupted. The return value is
    also saved as ``sequential_summary.json``.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress = {"status": "RUNNING", "stage": "stationary_refinement",
                "input_hashes": ctx.get("input_hashes", {}),
                "software": ctx.get("software", {}), "states": list(STATES),
                "saddles": [item[1] for item in EDGES],
                "source_sha256": sha256(__file__), "refined_points": [],
                "accepted_edges": []}
    write_json(out / "progress.json", progress)
    try:
        labels = [item for left, saddle, _ in EDGES for item in (left, saddle)] + [STATES[-1]]
        require(len(labels) == 13 and len(set(labels)) == 13,
                "Expected seven minima and six distinct saddle seeds")
        modes, stationary = {}, {}
        for label in labels:
            seed = np.asarray(ctx["points"][label]).copy()
            expected_energy = ctx["records"][label]["energy_kcal_mol"]
            x, mode, record = ctx["refine"](seed, int(label.startswith("TS_")))
            require(record["max_gradient_kcal_mol_A"] < GRADIENT_TOLERANCE,
                    label + " has excessive stationary-point residual force")
            require(abs(record["energy_kcal_mol"] - expected_energy) < 0.001,
                    label + " changed its reference energy unexpectedly")
            record.update(reference_energy_kcal_mol=expected_energy,
                          reference_energy_error_kcal_mol=record["energy_kcal_mol"] - expected_energy,
                          seed_displacement_max_atom_A=ctx["maxg"](x - seed))
            ctx["points"][label], ctx["records"][label] = x, record
            modes[label], stationary[label] = mode, record
            progress["refined_points"].append(label)
            write_json(out / "stationary_points.json", stationary)
            write_json(out / "progress.json", progress)
        edges, paths = {}, {}
        progress["stage"] = "downhill_connections"
        for left, label, right in EDGES:
            edges[label], paths[label] = _check_edge(ctx, out, left, label, right, modes[label])
            progress["accepted_edges"].append(label)
            write_json(out / "progress.json", progress)
        progress["stage"] = "whole_route_assembly"
        write_json(out / "progress.json", progress)
        trajectory = [ctx["points"]["A"].reshape(99, 3).copy()]
        joins, markers = [], [{"kind": "minimum", "label": "A", "input_image": 0}]
        for left, label, right in EDGES:
            segment = np.concatenate([paths[label][left][::-1], paths[label][right][1:]])
            fit = ctx["fit_transform"](trajectory[-1], segment[0])
            segment = (segment[:, fit["permutation"]] @ np.asarray(fit["rotation"]) +
                       np.asarray(fit["translation"]))
            gap = float(np.linalg.norm(segment[0] - trajectory[-1], axis=1).max())
            require(gap < ENDPOINT_TOLERANCE, "Whole-route segment join failed")
            offset = len(trajectory)
            joins.append({"left": left, "right": right, "saddle": label,
                          "fit_transform": fit, "fixed_for_entire_segment": True,
                          "maximum_atom_displacement_A": gap,
                          "first_input_image": offset,
                          "last_input_image": offset + len(segment) - 1})
            markers.extend([{"kind": "saddle", "label": label,
                             "input_image": offset + len(paths[label][left]) - 1},
                            {"kind": "minimum", "label": right,
                             "input_image": offset + len(segment) - 1}])
            trajectory.extend(segment)
        top = max(edge["saddle"]["energy_kcal_mol"] for edge in edges.values())
        xyz, energy, audit = audit_trace(ctx, trajectory, top,
                                         anchor_indices=[m["input_image"] for m in markers])
        require(audit["status"] == "VERIFIED_DISCRETE_PATH",
                "Joined sequential path failed its interval checks")
        for marker in markers:
            marker["image_index"] = audit["anchor_image_indices"][str(marker["input_image"])]
            marker["energy_kcal_mol"] = float(energy[marker["image_index"]])
        endpoint_fit = ctx["fit_transform"](xyz[-1], ctx["points"]["B"])
        require(endpoint_fit["all_atom_rmsd_A"] < ENDPOINT_TOLERANCE,
                "Joined route endpoint differs from B")
        a_energy = ctx["records"]["A"]["energy_kcal_mol"]
        torsions = np.asarray([angles(frame) for frame in xyz])
        path_file = out / "sequential_path.npz"
        np.savez_compressed(path_file, coordinates_angstrom=xyz, energy_kcal_mol=energy,
                            energy_above_A_kcal_mol=energy - a_energy,
                            torsions_deg=torsions,
                            unwrapped_torsions_deg=np.degrees(np.unwrap(np.radians(torsions), axis=0)),
                            fixed_atom_ids=np.arange(99), torsion_atom_indices=TORSIONS,
                            marker_image_indices=np.asarray([m["image_index"] for m in markers]))
        result = {"status": "SEQUENTIAL_ROUTE_REVALIDATED", "states": list(STATES),
                  "saddles": [item[1] for item in EDGES], "stationary_points": stationary,
                  "A_energy_kcal_mol": a_energy,
                  "highest_saddle_above_A_kcal_mol": top - a_energy,
                  "maximum_sampled_path_energy_above_A_kcal_mol":
                      audit["maximum_sampled_energy_kcal_mol"] - a_energy,
                  "path_check": audit, "joins": joins, "markers": markers,
                  "endpoint_B_fit": endpoint_fit, "edges": edges,
                  "path_file": path_file.name, "path_sha256": sha256(path_file),
                  "input_hashes": ctx.get("input_hashes", {}),
                  "software": ctx.get("software", {}), "source_sha256": sha256(__file__),
                  "elapsed_seconds": time.perf_counter() - started,
                  "labels": "P=N002 and Q=N015 in the exploratory archive. D and M retain their earlier meanings; D-M is a methyl rearrangement.",
                  "scope": "Revalidation of the selected route, not a complete network rediscovery, mass-weighted IRC, or global optimality proof. Finite fixed-atom gap checks do not certify a continuous supremum."}
        write_json(out / "sequential_summary.json", result)
        progress.update(status=result["status"], stage="complete", elapsed_seconds=result["elapsed_seconds"])
        write_json(out / "progress.json", progress)
        print("Sequential route: " +
              f"{result['maximum_sampled_path_energy_above_A_kcal_mol']:.6f} kcal/mol above A", flush=True)
        return result
    except BaseException as error:
        progress.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                        error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(out / "progress.json", progress)
        raise
