# Methods and scope

## Model and units

The molecule is an explicit 99-atom C43H56 graph, with 43 carbon atoms followed by 56 hydrogen atoms. The input graph, immutable atom order, and reference coordinates are packaged with the code. RDKit supplies the **MMFF94** energy and analytic Cartesian gradient. Energies are kcal/mol and coordinates are angstrom.

`nonBondedThresh=100.0` is the RDKit parameter used in constructing the force field. RDKit defines it as a multiple of a contact's minimum distance, not a literal 100 Å cutoff. The same model and parameter are used for every reported comparison. The force evaluator calls `CalcEnergy` before `CalcGrad` to refresh position-dependent state; a round-trip cache test and independent directional energy differences verify the evaluator.

The initial model is an independent reconstruction for the compound-6 question in Lindner, Grynszpan, and Biali (1993). It is not the original MM2 input file. The recorded local reference versions are Python 3.12.14, NumPy 2.2.6, SciPy 1.15.3, and RDKit 2025.9.6. Another version must pass current numerical checks before its output is accepted.

## Ordinary stationary points and symmetry

Six rigid translation/rotation directions are removed from the 297-dimensional Cartesian space, leaving 291 internal directions. A central finite difference of the analytic gradient gives the Hessian (standard step 0.0001 Å). The Hessian must be numerically symmetric before symmetrization. Ordinary endpoint minima have no negative internal curvature; accepted ordinary saddles have one. The mode-count threshold is 0.0001 in the corresponding Cartesian curvature units, and near-zero unresolved modes are rejected.

Stationary refinement uses a bounded local root solve. Acceptance requires a small actual gradient, the correct internal index, and consistency with the intended molecular basin. A solver flag is not sufficient.

Heavy-atom graph automorphisms are enumerated explicitly (32 for this graph). Basin matching uses a proper Kabsch rotation and graph-preserving assignment of identical hydrogens. No reflection is admitted in ordinary basin matching. A/B have a proper all-atom RMSD of approximately 1.91646 Å under that matching procedure; a separate diagnostic reflection makes them coincide to numerical precision. Thus enantiomer identity is not collapsed by alignment.

For path assembly, a selected graph map and proper rigid transform are applied once to an entire segment. Atoms are not independently relabeled from frame to frame. In the synchronized paths, all segment joins use identity atom ordering and proper rigid transforms only.

## Localized route

The selected route is A → C → P → Q → D → M → B. Its six saddles are TS_AC, TS_CP, TS_PQ, TS_QD, TS_DM, and TS_MB. Each seed is refined and classified. Small displacements in both signs of the unstable mode initiate energy-decreasing descent. Every accepted later minimization step is recorded and capped at a maximum atomic displacement of 0.020 Å.

The final root-refined endpoint is joined explicitly, matched to the expected basin, and included in the path audit. Whole segments are assembled in one consistent atom frame. Every retained interval is subdivided as necessary, and its quarter, half, and three-quarter points are evaluated. The maximum includes all recorded configurations, final connectors, inter-segment joins, and sampled gap energies.

This is a finite, numerically checked downhill/relaxation construction. It is not a mass-weighted intrinsic reaction coordinate, a thermal trajectory, or a rigorous analytic bound over every real-valued point of a continuous path.

## Exact collective progress

The six signed torsions use zero-based quadruples:

```text
(2,3,24,6)    (3,24,6,7)
(8,9,25,12)   (9,25,12,13)
(14,15,26,18) (15,26,18,19)
```

For each of two winding patterns, the progress constraint is

$$c_j(x;t)=\operatorname{wrap}\left[\phi_j(x)-a_j-t\Delta_j\right]=0,$$

with angles in radians, progress from 0 to 1, and the target changes taken from exact graph-equivalent representations of the B endpoint. The two conventions are called I/III and II/IV. Actual B torsions are used because a slightly nonplanar aromatic ring does not make every alternative ortho-atom dihedral differ by exactly 180°.

The optimization minimizes the **bare MMFF94 energy** under six angular equalities. A fixed invertible preconditioning chart in the 291-dimensional nonrigid Cartesian space improves conditioning. SLSQP proposes a point; a bounded KKT Newton solve can polish it. No penalty energy enters a quoted barrier.

The analytic torsion Jacobian is checked against independent central differences. The six constraints must have full rank. Multipliers are determined from the stationarity equations and checked directly, independently of any solver's success message or version-specific multiplier output.

With the convention

$$L(x,\lambda;t)=E(x)+\lambda^T c(x;t),$$

the actual KKT gradient is $\nabla E+J^T\lambda$. Accepted final branch minima require maximum atomic KKT-gradient norm at most $10^{-7}$ kcal/(mol Å), angular residual at most $10^{-10}$ rad, rank six, and positive constrained curvature. The projected energy gradient is also checked.

The appropriate constrained second derivative is

$$H_L=H_E+\sum_j\lambda_j H_{c_j}.$$

It is projected onto the **285-dimensional** tangent space after removing six rigid and six constraint directions. Using only the bare energy Hessian would be an incorrect minimum test at a constrained, nonstationary molecular geometry. An independent implementation, multiple finite-difference steps, and feasible energy-only second differences check this curvature calculation.

## Branches, connectors, and complete candidates

The initial study separately followed 41 nodes in each direction for each pattern: 164 converged constrained minima in total. Additional old-curve seeds and tighter tolerances test sensitivity. Forward/reverse discrepancies are retained as evidence of multiple local constrained conformations; they are not averaged away or spliced into a lower envelope.

Near the central changes, local bordered-KKT predictor/corrector steps follow individual stationary branches. When smaller steps cannot maintain residual and positive-curvature gates, the calculation records a continuation obstruction. That is evidence of a difficult or terminating branch; it is not, by itself, a proof of a mathematical fold.

Explicit fixed-progress connectors link the competing constrained minima. A constrained stationary seed must have exactly one negative **tangent** mode. Both signed releases are retracted onto the same six angular constraints, minimized by feasible energy-decreasing steps, and matched to the required constrained minima with fixed atom identities. The connector's own maximum energy enters the reported complete path height.

The current I/III candidate switches constrained conformations at progress 0.4675 and 0.525. The II/IV candidate uses 0.46 and 0.525. These switching progress values were selected in overlaps of observed local branches; they were not globally optimized over all possible synchronization definitions or all paths.

Outside the central switches, the construction uses explicitly retracted interpolation between converged nodes. The synchronized targets are satisfied to numerical precision at recorded configurations. Dense Cartesian interpolation between them has a small measured residual, recorded separately. Neither every interpolated point nor every point on a connector is a constrained local minimum.

The accepted candidates include physical connectors instead of discontinuous jumps. They are not globally minimized collective barriers. The finite gap audit is also not a proof of the continuous energy supremum. A tighter sampling repeat is used to assess the observed peak's sensitivity.

## Interpretation

The direct numerical claim is a comparison of three explicit complete paths in one model, with the same graph, endpoint basins, and energy reference. The localized route has lower maximum potential energy than the two synchronized candidates. It supplies numerical support for the localized mechanism in MMFF94.

The final checked heights above A are 7.4039947 kcal/mol for the localized route, 15.5659044 for I/III, and 15.5142011 for II/IV. The [final comparison record](../results/summary.json) retains their full precision and source files. The [I/III audit](../results/reference/collective/I_III/independent_audit.json) and [II/IV audit](../results/reference/collective/II_IV/independent_audit.json) both passed. These complete-path values include the selected constrained bridges and peripheral interpolation; they are distinct from maxima over the four 41-node sequences alone.

No rate follows from these heights alone. Free energies, entropy, temperature, solvent, and kinetic network effects have not been computed. The MM2 result in the historical paper is a different calculation. The paper's experimental 7.4 kcal/mol bound refers to compound 5, not the compound 6 modeled here.

## Primary references

- [Lindner, Grynszpan, and Biali (1993)](https://doi.org/10.1021/jo00076a027).
- [RDKit force-field API](https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html).
- [RDKit MMFF construction and threshold semantics](https://www.rdkit.org/docs/cppapi/namespaceRDKit_1_1MMFF.html).
- [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html). Actual residual and curvature checks in this project are separate from the solver's termination criteria.
