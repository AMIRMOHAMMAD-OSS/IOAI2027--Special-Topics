# Reproducing the calculations

There are two entry points:

| Starting point | Workflow |
|---|---|
| A source checkout of this repository | Use the packaged CLI to verify the model, revalidate the localized route, study constrained nodes, or reconstruct the complete synchronized candidates. |
| The existing validated Mac Studio calculation | Paste a readable Python file into nano for the exact-constraint node study; after it passes, use the separate single-file complete-path continuation. |

The packaged workflow is self-contained. The Mac continuation deliberately uses the already validated local files. All quoted energies are bare MMFF94 potential energies in kcal/mol, with coordinates in angstrom.

## 1. A clean source checkout

First locate the source checkout. Its top-level folder contains `pyproject.toml`, `requirements-reproduce.txt`, `README.md`, and `src/`. Open a terminal there. On a Mac, you can type `cd `, drag that actual folder from Finder into Terminal, and press Enter. Use the folder you actually obtained; creating an empty directory with the project name does not install the source.

The recorded local reference environment used **Python 3.12.14**, NumPy **2.2.6**, SciPy **1.15.3**, and RDKit **2025.09.6**. The Python package pins are in `requirements-reproduce.txt`. With a Python 3.12 interpreter available, run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps -e .
python -m propeller_paths verify
```

The first installation needs access to the package index. The calculations use the files in the repository. No GPU is required. Other compatible Python/library versions must pass the numerical checks; matching an environment name alone does not establish reproducibility.

### Choose the calculation

Every command below first checks the explicit molecular model, analytical derivatives, enantiomer distinction, and all 13 saved stationary points. By default, a run creates a new timestamped folder under `runs/` and prints its absolute location; `--out` can select another new destination.

| Command | Calculation |
|---|---|
| `python -m propeller_paths verify` | Check the explicit graph, MMFF94 calculator, derivatives, proper endpoint matching, and saved stationary-point indices. |
| `python -m propeller_paths sequential` | Refine the 13 seeds and revalidate the six saddle connections in A → C → P → Q → D → M → B, including the joined path. |
| `python -m propeller_paths collective` | Compute both synchronized patterns in both directions: 40 intervals, 41 nodes per branch, **164 constrained nodes** in total. Audit the proposed node-to-node chords separately. |
| `python -m propeller_paths paths` | Reconstruct **both complete synchronized candidates**, including their central branch connections, endpoint joins, projected connecting segments, and dense finite checks. |
| `python -m propeller_paths reproduce` | Revalidate the selected localized route and reconstruct both complete synchronized candidates in one run. |

`collective` studies **node sequences**. Its node maxima do not include every possible barrier between those nodes. The `paths` and `reproduce` commands include the explicit connectors used in the reported complete candidates. The combined `reproduce` command does not also rerun the separate four-direction node survey or the earlier exploratory transition-network search.

The reference complete-path heights above A are approximately:

| Candidate | Checked height above A |
|---|---:|
| Localized A → C → P → Q → D → M → B | 7.4039947 kcal/mol |
| Synchronized I/III | 15.5659044 kcal/mol |
| Synchronized II/IV | 15.5142011 kcal/mol |

Both synchronized candidates passed separate numerical audits. The [final comparison record](../results/summary.json) links their source data and audit reports. These values describe the constructed paths. Acceptance depends on the saved numerical checks, not on reproducing a rounded number alone. The calculations do not prove global optimality or predict experimental rates.

### Optional controls

Use `--out` to choose a **new** output folder:

```bash
python -m propeller_paths sequential --out runs/my_sequential_check
```

An existing output directory is rejected. For another run, choose another name or omit `--out` to obtain a fresh timestamped directory.

A short node-only trial uses:

```bash
python -m propeller_paths collective --intervals 4 --pattern I_III --direction forward
```

That computes five constrained nodes. It exercises the numerical workflow; it does not reproduce the complete synchronized path or its final height.

| Option | Applies to |
|---|---|
| `--pattern I_III` or `--pattern II_IV` | Select one synchronized pattern for `collective`, `paths`, or `reproduce`. Default: both. |
| `--direction forward` or `--direction reverse` | The `collective` node study only. Default: both. |
| `--intervals N` | The `collective` node study only; at least four intervals. |
| `--gradient-tol`, `--constraint-tol` | The `collective` node study only. Defaults: `1e-7` and `1e-10` rad. |
| `--atom-step VALUE` | Dense complete-path construction for `paths`/`reproduce`; positive and no greater than `0.01` Å. |
| `--no-chord-audit` | Skip the separate chord diagnostics for `collective`; this does not confer continuous-path validation. |

The complete-path workflow uses its checked 40-interval starting construction and internal continuation settings. Passing `--intervals 4` does not make `paths` a short four-interval smoke test.

### Read the reports

| Output | Contents |
|---|---|
| `run.json` | Command, arguments, software versions, input hashes, status, result locations, and elapsed time. |
| `model_checks.json` | Calculator consistency, derivative, chirality, energy, gradient, and stationary-index checks. |
| `sequential/sequential_summary.json` | Refined points, six accepted connections, whole-route joins, exact frame markers, and sampled maximum. |
| `sequential/sequential_path.npz` | Full joined coordinates, energies, torsions, atom IDs, and marker indices. Raw branch traces are saved alongside it. |
| `I_III_forward/branch.json` and analogous branch folders | Node statuses, residuals, curvature, endpoint matches, and node-energy maxima. Each node has its own result and optimizer trace. |
| Each node branch’s `chord_audit.json` | Energy and constraint diagnostics for proposed direct connectors; these are not automatically accepted paths. |
| `I_III_paths/complete_path/summary.json` and the II/IV equivalent | Complete candidate status, source joins, endpoint checks, finite maximum energy, constraints, and interval bounds. |
| Each complete path’s `trajectory.npz` | Recorded coordinates, progress, and energies for the reconstructed candidate. |
| Each complete-path construction’s `left_connector/summary.json` and `right_connector/summary.json` | Constrained saddle, both feasible descents, endpoint assignments, and connector checks. |

A successful CLI run ends with `REQUESTED_CHECKS_COMPLETED`. Also inspect the nested scientific status: `SEQUENTIAL_ROUTE_REVALIDATED`, `COMPLETE_CONSTRAINED_NODE_SEQUENCE`, or `COMPLETE_EXACT_FEASIBLE_COLLECTIVE_PATH`, according to the command. A stop retains diagnostic output; a partially completed report is not an accepted result.

### Numerical gates

| Stage | Principal acceptance criteria |
|---|---|
| Saved-point verification | Reference energy agreement within `0.001` kcal/mol, maximum atomic gradient norm below `1e-4` kcal/mol/Å, and the required resolved internal Hessian index. |
| Refined localized route | Maximum atomic gradient norm below `1e-5` kcal/mol/Å; minima have index zero and saddles index one; proper endpoint matching and final connectors are checked. |
| Localized path intervals | Maximum checked atomic displacement `0.020` Å; quarter, midpoint, and three-quarter energies are probed. Allowed sampled excess above the saddle is `1e-5` kcal/mol and is included in the reported sampled maximum. |
| Constrained minimum nodes | Six independent angular equalities, maximum residual at most `1e-10` rad, KKT/projected atomic-gradient norm at most `1e-7`, and lowest tangent curvature above `1e-4`. |
| Complete synchronized path | Stored-node angular residual below `1e-8` rad; chord-probe residual below `1e-4` rad; small atomic gaps; intended stationary endpoints; all source joins and connector checks accepted. |

Constrained curvature uses the **Lagrangian Hessian** on the constraint tangent space. The fixed-progress connector saddles are constrained saddles, not ordinary unrestricted transition states. The complete path’s connecting frames are explicitly projected and checked; they are not all constrained energy minima. All interval checks are finite numerical evidence, not a mathematical certificate of the continuous energy supremum.

### Tests used by CI

From the installed checkout, run:

```bash
python -m unittest discover -s tests -v
```

All five tests passed in the recorded local check. The GitHub Actions workflow is configured but has not run on GitHub; the recorded pass is local. The suite covers derivatives, proper chirality/endpoint matching, recorded capped steps, and detection of a hidden barrier between low-energy endpoints. It uses Python’s standard-library unittest runner; pytest is not needed for this command. This short suite does not substitute for the full `sequential` or `paths` calculations.

## 2. Continue the existing Mac Studio calculation with one readable file

This workflow uses the previously validated Mac folder:

```text
/Volumes/LaCie/riemannian_propellers/lindner_readable.72dGzI2Z/comparison_results_1794512270493666000
```

The first continuation runs the exact-constraint **node-convergence** study. After that study passes, the complete-path continuation below reuses its accepted nodes and reconstructs the explicit synchronized connectors. The packaged `paths` command remains the independent clean-checkout entry point.

Both stages have now been reported successful on the user's Mac Studio. The [reported complete-path summary](../results/reported_mac/complete_path_summary.json) is preserved separately from the local reference datasets; the full Mac result directory has not been retrieved. The instructions below describe how to repeat those stages.

### Create the file in nano

1. Open the complete readable source of [`tools/lindner_exact_collective.py`](../tools/lindner_exact_collective.py), or use the complete source supplied with the project. Copy the entire file: its first line is `#!/usr/bin/env python3` and its final line is `# END_LINDNER_EXACT_COLLECTIVE`. The corrected version has 361 lines.
2. In the **Mac Studio terminal**, open the editor:

   ```bash
   nano "$HOME/lindner_exact_collective.py"
   ```

3. Paste the complete Python source into nano’s editing area. If an older copy is already open, replace its contents first. Copy the source text itself, without Markdown code-fence lines.
4. Save with **Control+O**, press **Enter** to confirm the filename, then leave nano with **Control+X**.
5. Run in the same Mac terminal:

   ```bash
   conda activate neural_opt
   python -u "$HOME/lindner_exact_collective.py"
   ```

The script includes its own checked input loader. It verifies the original central `revalidate.py` hash before extracting the required model definitions, selects the passed terminal report whose hash is recorded in this comparison, checks central/terminal/reference provenance, and validates current energies, gradients, chirality, A/B endpoints, and winding targets. It does not import the saved `runner_source.py`. It uses the existing NumPy, SciPy, and RDKit environment and creates a fresh `exact_collective_...` subfolder inside the comparison folder.

### Repair a previously copied continuation

The first delivered continuation required the entire saved comparison runner to match the assistant's local source bytes. The actual Mac runner was never byte-inspected; a successful earlier calculation did not establish that whole-file identity. A reported mismatch stopped before any new calculation or output change. The corrected loader removes that unnecessary dependency while retaining the original central-source and numerical checks.

For the existing `~/lindner_exact_collective2.py`, paste [`tools/lindner_repair_collective.py`](../tools/lindner_repair_collective.py) into a single new file:

```bash
nano -w "$HOME/lindner_repair_collective.py"
```

Save the complete source and run:

```bash
conda activate neural_opt
python -u "$HOME/lindner_repair_collective.py"
```

The repair checks the syntax tree of the supplied continuation, tolerating comments, whitespace, and Python 3.11/3.12's empty `type_params` difference. It replaces the input loader and associated provenance metadata, verifies that the new file compiles, writes a new uniquely named Python file, and runs it. The existing script and saved results are preserved. The numerical solver is unchanged. A different original filename can be selected with `--source /actual/path/to/script.py`; remaining arguments, such as `--intervals 4`, are passed to the repaired continuation.

By default it computes **I/III forward, I/III reverse, II/IV forward, and II/IV reverse**, with **41 nodes each**. Each printed node includes its relative energy, KKT residual, and acceptance status. Successful completion ends with:

```text
COMPLETE_EXACT_CONSTRAINED_NODE_SEQUENCES
```

This status establishes accepted nodes and final endpoint matches. The file explicitly reports that connecting segments and their energies need separate validation; a node maximum is not a complete-path maximum.

### A shorter Mac workflow check

Run both patterns and both directions on a five-node grid:

```bash
python -u "$HOME/lindner_exact_collective.py" --intervals 4
```

For only one pattern on that grid:

```bash
python -u "$HOME/lindner_exact_collective.py" --intervals 4 --pattern I_III
```

Each invocation creates a fresh result folder. If the existing validated comparison has a different location, `--comparison` accepts its actual path. `--out` accepts a new output location. These options select existing inputs and a new destination; they do not construct missing prior validation files.

The output contains `summary.json`, a copy of the executed source, and one folder per pattern/direction. Each branch contains `branch.json`, `nodes.npz`, and individual node records with optimizer traces. Numerical failures are reported as unresolved node sequences, with diagnostic files retained.

For maintainers, the embedded solver can be checked against the package module from a source checkout:

```bash
python tools/build_lindner_exact_collective.py --check
```

### Continue from accepted Mac nodes to complete candidates

The complete-path program is [`tools/lindner_complete_paths.py`](../tools/lindner_complete_paths.py). It uses the completed 41-node study for both patterns and directions. Its default input is the specific Mac result folder reported in this session; another result folder can be selected with `--nodes /actual/path/to/exact_collective_run`.

Paste the complete readable source into a new file:

```bash
nano -w "$HOME/lindner_complete_paths.py"
```

Save with Control+O, Enter, and Control+X, then run:

```bash
conda activate neural_opt
python -u "$HOME/lindner_complete_paths.py"
```

The final source line is `# END_LINDNER_COMPLETE_PATHS`. The program creates a new timestamped directory inside the existing node result directory. It reads the original result files and leaves their contents unchanged. It uses the existing NumPy, SciPy, and RDKit environment.

Before construction, the continuation checks the saved source hash and the portable syntax-tree identity of the corrected node solver. It tolerates comments, whitespace, and the empty Python 3.12 `type_params` field while rejecting substantive source changes. It then reruns the original model/provenance checks, validates the saved node reports and arrays, recomputes forces and angular constraints for each reused forward node, and checks tangent curvature at the branch starting points. This avoids depending on an exact byte copy of the older `runner_source.py`.

For each pattern it continues four local constrained-minimum branches from progress 0.45, 0.50, and 0.55, refines two recorded constrained-saddle seeds, and follows both feasible downhill directions to verify their endpoint assignments. It then joins the blocks using fixed atom identities and proper rigid rotations, projects the connecting segments, and checks every stored frame plus quarter, midpoint, and three-quarter chord probes. The supplied connector coordinates are starting guesses for revalidation; this continuation does not claim an independent saddle search.

A successful run ends with `COMPLETE_COLLECTIVE_CANDIDATE_PATHS_VERIFIED`. The printed summary gives the selected synchronized candidate heights and their comparison with the previously checked sequential path. Per-pattern `complete_path/trajectory.npz` and `complete_path/summary.json` preserve the full constructed path and checks; branch, constrained-saddle, and release records are retained alongside them. The top-level `summary.json` records inputs, source hashes, software provenance, and status. A stopped run retains diagnostics without counting a partial path as accepted.

The constraints are exact to numerical tolerance at the stored projected frames. Chord probes are finite checks, and peripheral interpolation is not fully minimized. These outputs establish explicit checked candidates; they do not certify the continuous energy supremum, global collective optimality, or experimental rates.

## 3. What has actually been run

| Recorded local validation | Observed calculation time | Scope |
|---|---:|---|
| Model and saved-point verification | About 9.8 s | Original recorded verification, including the 13 stationary points. |
| Full standalone sequential revalidation | 224.69 s | Thirteen refinements, six saddles, twelve branches, joins, and a 16,343-frame whole path. The CLI adds its initial model verification. |
| Original single-file Mac continuation, before the loader correction, exercised against the local Mac-layout fixture | 98.69 s | Both patterns and directions with four intervals: 20 accepted constrained nodes. |
| Corrected loader and single-file repair, exercised with an invalid saved runner | 46.36 s | I/III in both directions with four intervals: 10 accepted nodes; the complete 15-check regression passed. |
| Single-file complete-path continuation, using a local fixture built from the 164 reference nodes | 175.44 s | Both complete synchronized candidates, four constrained connectors, and eight feasible downhill releases passed. An independent audit checked all 3,115 stored frames and 9,339 additional chord probes. |

The four default 41-node sequences were separately calculated and checked in the local study, giving 164 accepted nodes. The original continuation's direct smoke test used the smaller 20-node grid; the loader correction was exercised through the repair file on a 10-node grid. All these tests ran locally, not remotely on the user's Mac. The timings above are observations from the recorded local environment; processor, library, threading, and convergence differences affect runtime on other machines. The loader correction's full checks and source hashes are in [`mac_loader_fix_regression.json`](../results/reference/test_records/mac_loader_fix_regression.json). Its Python 3.11 AST compatibility check simulated the absence of empty `type_params` fields; the executed chemistry smoke used Python 3.12.14.

The complete-path fixture reuses the local reference node coordinates and records in the Mac continuation's directory schema. It is explicitly labeled as a reconstructed local fixture, not as files retrieved from the Mac. The user's earlier Mac node-study output displayed 22 II/IV reverse-node energies that match the local reference at the shown six decimal places. The subsequent complete-path result is recorded below.

The [complete-path continuation test record](../results/reference/test_records/mac_complete_paths_regression.json) contains all 22 final guard checks and the independent saved-path audit. The captured full-run source differs from the delivered source only in strengthened input validation and compact reporting; all 11 numerical/construction functions and four connector seeds are unchanged. The final input guards and summary formatter were executed separately after those edits. The complete local test trajectories, releases, branches, captured source, and audit records are retained under [`mac_complete_paths_validation/`](../results/reference/collective/mac_complete_paths_validation/).

Historical guard tests of the original node continuation rejected a changed saved runner. The corrected node loader deliberately removes that unnecessary dependency while checking the original model and result provenance. The complete-path continuation additionally checks the recorded node-source hash and its portable executable syntax-tree identity. The source files and scientific checks, rather than elapsed time or a success message alone, establish what each run completed.

### Reported completed Mac Studio run

The user ran the single-file complete-path continuation on the Mac and pasted its final summary. It reported `COMPLETE_COLLECTIVE_CANDIDATE_PATHS_VERIFIED`, with **all four constrained connectors verified** and both candidate paths passing their numerical gates.

| Quantity reported by the Mac | Height above A, kcal/mol | Stored frames |
|---|---:|---:|
| Previously validated localized path baseline | 7.403994698690468 | Reused from the earlier run |
| Complete synchronized I/III candidate | 15.565904425506801 | 1,527 |
| Complete synchronized II/IV candidate | 15.514201146247729 | 1,586 |

The reported elapsed time was **43.918 s** for this continuation. It reused the accepted node study and the earlier sequential path record while rechecking the six sequential saddle energies and forces; this time does not include those earlier calculations or a new sequential-path reconstruction.

The [reported terminal summary](../results/reported_mac/complete_path_summary.json) retains the user-supplied output, and [comparison_with_local.json](../results/reported_mac/comparison_with_local.json) records its comparison with the local reference. These are receipts of pasted terminal output. The raw Mac trajectories, per-node files, and connector reports have not been accessed or uploaded, so no independent audit of those Mac files is claimed. The complete local datasets and their independent audits remain unchanged under `results/reference/`.

This reported reproduction agrees with the comparison of the specified candidate paths. It does not extend the result to every collective route, establish a globally minimal path, or make all interpolated configurations constrained minima.
