"""Command-line reproduction; every run writes into a new directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from .core import DATA, load_context, verify_core, write_json, software_versions, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(prog="propeller-paths",
        description="Reproduce the explicit MMFF94 propeller-chain calculations.")
    parser.add_argument("command", choices=("verify", "sequential", "collective", "paths", "reproduce"))
    parser.add_argument("--out", type=Path, help="New directory; existing directories are rejected")
    parser.add_argument("--intervals", type=int, default=40, help="Node intervals for collective only (default 40); paths uses the verified 40-interval workflow")
    parser.add_argument("--pattern", choices=("both", "I_III", "II_IV"), default="both")
    parser.add_argument("--direction", choices=("both", "forward", "reverse"), default="both",
                        help="Continuation direction for the collective node-only command")
    parser.add_argument("--gradient-tol", type=float, default=1e-7,
                        help="KKT tolerance for collective only; paths uses 1e-7")
    parser.add_argument("--constraint-tol", type=float, default=1e-10,
                        help="Angular tolerance for collective only; paths uses 1e-10 rad")
    parser.add_argument("--atom-step", type=float, default=0.01,
                        help="Dense complete-path atom step in angstrom, 0 < step <= 0.01 (paths/reproduce)")
    parser.add_argument("--no-chord-audit", action="store_true",
                        help="Skip separate node-connector diagnostics; does not confer path validation")
    args = parser.parse_args(argv)
    if args.intervals < 4 or not (args.gradient_tol > 0 and args.constraint_tol > 0):
        parser.error("Positive tolerances and at least four intervals are required")
    if not (0 < args.atom_step <= 0.01):
        parser.error("--atom-step must be positive and no greater than 0.01 angstrom")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.out if args.out is not None else Path("runs") / (args.command + "_" + stamp)
    if out.exists():
        parser.error("Output already exists; choose a new directory: " + str(out))
    out.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "RUNNING", "command": args.command, "arguments": vars(args),
              "software": software_versions(), "started_utc": stamp, "output": str(out.resolve())}
    write_json(out / "run.json", report)
    print("New results:", out.resolve(), flush=True)
    try:
        ctx = load_context()
        report["input_hashes"] = ctx["input_hashes"]
        checks = verify_core(ctx, stationary=True)
        write_json(out / "model_checks.json", checks)
        print("Explicit model, derivatives, enantiomers, and all 13 stationary points verified.", flush=True)
        if args.command in ("sequential", "reproduce"):
            from .sequential import run_sequential
            sequential = run_sequential(ctx, out / "sequential")
            report["sequential"] = {
                "status": sequential["status"],
                "maximum_sampled_path_energy_above_A_kcal_mol": sequential["maximum_sampled_path_energy_above_A_kcal_mol"]}
            write_json(out / "run.json", report)
        if args.command == "collective":
            from .collective import run_branch
            patterns = ("I_III", "II_IV") if args.pattern == "both" else (args.pattern,)
            directions = ("forward", "reverse") if args.direction == "both" else (args.direction,)
            report["collective_nodes"] = []
            for pattern in patterns:
                for direction in directions:
                    branch = run_branch(ctx, pattern, direction, out / (pattern + "_" + direction),
                                        args.intervals, args.gradient_tol, args.constraint_tol,
                                        audit=not args.no_chord_audit)
                    report["collective_nodes"].append({
                        "pattern": pattern, "direction": direction, "status": branch["status"],
                        "maximum_node_energy_above_A_kcal_mol": branch["maximum_node_energy_above_A_kcal_mol"]})
                    write_json(out / "run.json", report)
            report["collective_scope"] = "Converged constrained node sequences. These maxima are not complete-path maxima."
        if args.command in ("paths", "reproduce"):
            from .collective_paths import complete_collective_path
            patterns = ("I_III", "II_IV") if args.pattern == "both" else (args.pattern,)
            seed_file = DATA / "collective_connector_seeds.json"
            report["collective_connector_seeds_sha256"] = sha256(seed_file)
            report["synchronized_paths"] = []
            report["synchronized_path_settings"] = {
                "forward_node_intervals": 40, "constraint_tolerance_rad": 1e-10,
                "kkt_gradient_tolerance": 1e-7, "atom_step_A": args.atom_step,
                "connector_seed_source": "packaged quantized constrained saddle seeds"}
            for pattern in patterns:
                path_out = out / (pattern + "_paths")
                result = complete_collective_path(ctx, pattern, path_out, seed_file,
                                                  atom_step=args.atom_step)
                report["synchronized_paths"].append({
                    "pattern": pattern, "status": result["status"],
                    "maximum_finite_checked_energy_kcal_mol": result["maximum_finite_checked_energy_kcal_mol"],
                    "finite_checked_height_above_A_kcal_mol": result["finite_checked_height_above_A_kcal_mol"],
                    "frames": result["frames"],
                    "workflow_elapsed_seconds": result["complete_workflow_elapsed_seconds"],
                    "maximum_node_constraint_rad": result["maximum_node_constraint_rad"],
                    "maximum_chord_probe_constraint_rad": result["maximum_chord_probe_constraint_rad"],
                    "maximum_atom_step_A": result["maximum_atom_step_A"],
                    "checks": result["checks"],
                    "result_directory": str((path_out / "complete_path").resolve())})
                write_json(out / "run.json", report)
            report["synchronized_path_scope"] = (
                "Complete explicit candidate paths with converged constrained branch nodes, "
                "verified fixed-progress saddle/downhill connectors, exact-feasible sampled nodes, "
                "and finite chord checks. These are not proven globally minimum collective barriers.")
        report["status"] = "REQUESTED_CHECKS_COMPLETED"
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "run.json", report)
        print(json.dumps({k: v for k, v in report.items()
                          if k in ("status", "sequential", "collective_nodes", "collective_scope", "synchronized_paths", "synchronized_path_scope", "elapsed_seconds", "output")}, indent=2), flush=True)
        return 0
    except Exception as error:
        report.update(status="STOPPED_WITH_SAVED_PROGRESS", error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        write_json(out / "run.json", report)
        print("Stopped:", error, "\nSaved progress:", out.resolve(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
