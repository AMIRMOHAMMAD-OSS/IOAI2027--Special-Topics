#!/usr/bin/env python3
"""Refresh the standalone Mac file's solver from the maintained package module."""
import argparse
import hashlib
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/propeller_paths/constraints.py").read_text()
    target = root / "tools/lindner_exact_collective.py"
    old = target.read_text()
    begin, end = "# BEGIN_EMBEDDED_CONSTRAINT_SOLVER", "# END_EMBEDDED_CONSTRAINT_SOLVER"
    prefix, rest = old.split(begin, 1)
    _, suffix = rest.split(end, 1)
    payload = source[source.index("TORSIONS="):].rstrip()
    output = prefix + begin + "\n" + payload + "\n" + end + suffix
    digest = hashlib.sha256(source.encode()).hexdigest()
    output = re.sub(r'EMBEDDED_SOLVER_SHA256 = "[^"]+"',
                    'EMBEDDED_SOLVER_SHA256 = "' + digest + '"', output, count=1)
    compile(output, str(target), "exec")
    if not output.rstrip().endswith("# END_LINDNER_EXACT_COLLECTIVE"):
        raise SystemExit("Missing completion marker")
    if args.check and old != output:
        raise SystemExit("The standalone solver is stale; run the builder without --check.")
    if not args.check:
        target.write_text(output)
    print("Standalone solver matches constraints.py; SHA-256 " + digest)


if __name__ == "__main__":
    main()
