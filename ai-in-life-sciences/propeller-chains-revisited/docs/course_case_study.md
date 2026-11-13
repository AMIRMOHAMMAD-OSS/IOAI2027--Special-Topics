# Teaching case: geometry-aware proposals for molecular motion

**Course context.** This is a small illustrative use case for the instructor’s planned **AI in Life Sciences** course for the **National Olympiad on Artificial Intelligence committee**, preparing students for **IOAI 2027**. The case connects chemical geometry to the design and evaluation of generative models.

## Learning objectives

By the end, students should be able to:

- Explain why plausible molecular endpoints do not guarantee a plausible connecting motion.
- Distinguish rigid rotations, identical-atom permutations, and reflections when comparing structures.
- Separate a generated proposal from an energetically and geometrically checked result.
- Design an evaluation that checks both individual conformers and complete paths.

## The chemical question

A four-ring molecular chain can change handedness through coordinated rotations. This project revisits the question studied by Lindner, Grynszpan, and Biali using an explicit **99-atom MMFF94 model**. It constructs a localized route and two prescribed synchronized candidates. Their finite numerically checked heights above the same endpoint A are approximately **7.404, 15.566, and 15.514 kcal/mol**, respectively. These are potential-energy comparisons in this model; the original paper used MM2. See the [comparison record](../results/summary.json) and [methods](methods.md).

Start with the [recorded animation](../figures/localized_motion.gif) or [interactive explorer](explorer.html). Ask: “What would an algorithm need to understand to propose this motion?” The animation’s playback speed has no physical time interpretation.

## Coordinates, constraints, and symmetry

Cartesian coordinates describe where every atom is. Straight interpolation between two coordinate arrays moves each atom along a chord. A rotation instead follows an arc. For a simple bond of length $r$, two orientations separated by an angle $\theta<\pi$ have a Cartesian midpoint only $r\cos(\theta/2)$ from the pivot. Thus endpoint bond lengths alone do not ensure intermediate bond lengths.

Torsions describe rotation more directly. Here six angular constraints prescribe collective progress while the remaining coordinates relax. Those equalities encode the chosen synchronized motion; general chemical validity also requires checking bond geometry and nonbonded contacts. Torsional coordinates still need energy and geometry checks: specifying angles does not automatically prevent steric clashes or find the lowest-energy route. The [methods](methods.md#exact-collective-progress) define the constraints and convergence tests.

Identical atoms require another distinction. Exchanging equivalent methyl hydrogens can create duplicate representations of one physical conformation. Matching must respect the molecular graph. Reflections require different treatment: they can exchange the A/B enantiomers that this study deliberately distinguishes. During path assembly, one consistent atom correspondence is maintained across each segment; independently relabeling every frame could conceal jumps. See [symmetry and matching](methods.md#ordinary-stationary-points-and-symmetry).

## From a generative proposal to evidence

**Conformer generation** proposes individual structures. **Path generation** proposes an ordered connection between specified structures. A set of good conformers can still have poor connections.

Use the following evaluation sequence for a future model:

1. **Proposal:** generate coordinates, torsions, or a sequence of configurations under a stated endpoint condition.
2. **Geometry and constraints:** verify atom identity, graph consistency, finite coordinates, bond geometry, and any prescribed torsions.
3. **Energy and stationarity:** evaluate the same force field; refine where appropriate and inspect actual gradient and curvature conditions.
4. **Continuity:** inspect connecting intervals, including energy between saved configurations. Retain branch changes and their connectors.
5. **Connectivity:** verify that the complete curve reaches the intended basins. For an ordinary transition-state claim, verify the saddle index and its two downhill endpoints.

In the existing calculation, **164 constrained minima** were accepted, but their node maxima were insufficient to characterize complete paths. Four fixed-progress constrained saddle connectors supplied the central branch connections. Those constrained saddles are distinct from ordinary unrestricted transition states. Peripheral interpolation is checked but not fully minimized. [Evidence and scope](../results/summary.json); [construction](methods.md#branches-connectors-and-complete-candidates).

## Three small exercises

**1. Find a hidden interpolation cost.** Inspect the entry with `from_node: 21`, from progress 0.525 to 0.55, in the [I/III coarse-chord diagnostic](../results/reference/collective/rejected_coarse_chord_diagnostics/I_III_forward_40.json). Compare endpoint energies with the midpoint probe.

**Expected observation:** the midpoint reaches approximately **25.707 kcal/mol above A**, although the endpoint heights are approximately **15.375 and 14.983**. The largest atomic displacement across this interval is about **1.328 Å**. A plot of node energies alone misses this costly proposed connection. This diagnoses that interpolation, not every possible Cartesian method.

**2. Choose the correct equivalence rule.** Read the proper and reflected A/B RMSDs in the [model checks](../results/reference/test_records/model_checks.json). Should a generator’s uniqueness metric count A and B as one conformer?

**Expected observation:** proper matching leaves an RMSD of about **1.916 Å**; the diagnostic reflection reduces it to numerical zero. A/B remain distinct enantiomers for this task. Equivalent-atom permutations and reflections cannot be treated interchangeably.

**3. Write a defensible result sentence.** Use the three heights in the [comparison](../results/summary.json), calculate the synchronized-minus-localized differences, and decide what follows.

**Expected observation:** the differences are approximately **8.162 and 8.110 kcal/mol**. The supported statement compares these three explicit candidates. It does not establish that every synchronized route is higher, a globally optimal mechanism, or an experimental rate.

## Optional advanced extension

In local nonrigid coordinates, let $J$ be a full-row-rank constraint Jacobian and $g$ the energy gradient. The projector

$$
P=I-J^T(JJ^T)^{-1}J
$$

gives a first-order feasible direction $-Pg$. Curved constraints still require retraction after a finite step. A small projected gradient indicates constrained stationarity; classifying a minimum additionally requires the tangent-space **Lagrangian Hessian**. [Implemented checks](methods.md#exact-collective-progress).

## Future generative evaluation

Compare raw proposals and refined outputs separately. Useful metrics include geometric validity, graph-aware uniqueness that preserves chirality, coverage of reference basins, endpoint-connection success, constraint residuals, maximum checked path energy, refinement displacement, and force-field evaluations per accepted path. Keep endpoints, force field, and evaluation budgets consistent; retain failed proposals.

**No generative model is trained or evaluated here.** The current repository supplies reference structures, checked candidate paths, and instructive failure cases for a future controlled comparison. Its computational timings are not evidence of generative-model performance. See the [claim ledger](claims.md).
