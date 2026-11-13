# Claim ledger

This ledger separates the historical source, the present numerical records, interpretation, and future work. Page numbers refer to the printed journal pages in Lindner, Grynszpan, and Biali (1993), DOI [10.1021/jo00076a027](https://doi.org/10.1021/jo00076a027).

## Historical paper

| ID | Claim supported by the source | Evidence | Safe use / qualification |
|---|---|---|---|
| P01 | The first author is Ariel B. Lindner; coauthors are Flavio Grynszpan and Silvio E. Biali. | PDF p.6662 and official ACS volume 58, issue 24 table of contents. | Do not use Andreas. The source gives the middle initial B.; do not expand it. |
| P02 | The paper appeared in J. Org. Chem. 58(24), 6662–6670 (1993). | PDF masthead and ACS TOC, issue date November 1, 1993. | Received June 25 and Advance ACS abstract October 1 are different dates. PDF-file creation metadata is not publication metadata. |
| P03 | Systems 5 and 6 contain two and three overlapping diarylmethyl moieties, respectively. | Abstract p.6662; structures p.6663. | Compound 6 has four aromatic rings; do not call it a three-ring molecule because it has three propeller subunits. |
| P04 | The calculations used MM2 in BIGSTRN-3. | Computational methods p.6664. | Our RDKit MMFF94 calculation is a separate model test. No numerical MM2 reproduction has been completed. |
| P05 | Compound 6 omitted terminal para substituents to stay within the program’s 100-atom limit. | Footnote 10, p.6663. | Useful historical motivation for the specific graph; it does not justify deleting atoms from our explicit model. |
| P06 | The paper synthesized and performed variable-temperature NMR on compound 5. | pp.6664, 6670. | Do not imply that the experimental work measured the compound-6 barrier. |
| P07 | The 7.4 kcal/mol NMR estimate is an upper bound for compound 5 under stated assumptions. | p.6664, discussion revisited p.6668. | It is not a measured compound-6 potential barrier and does not validate our numerically similar 7.40399 result. |
| P08 | The lowest ideal collective flip for compound 6 was estimated at 10 kcal/mol. | Table IV and discussion p.6669. | The ideal structures froze ring torsions at combinations of 0° and 90°. This is not a modern unconstrained index-one saddle certificate. |
| P09 | Localized compound-6 barriers were also estimated with the two rotating torsions fixed at 0° and 90°. | Footnote 23, p.6669. | Compare the qualitative mechanism while identifying the different optimization/validation methods. |
| P10 | The paper’s coarse graph has 20 conformers / 10 enantiomeric pairs. | Table II, pp.6665–6666; Figure 10, p.6669. | A coarse form count does not enumerate all methyl rotamers or all microscopic stationary points. |
| P11 | A three-local-event enantiomerization route has a 5.5 kcal/mol bottleneck, and a longer route has 4.3 kcal/mol. | pp.6669–6670; Figure 10. | These are graph barriers relative to conformer c in MM2. They are not directional barriers measured from each preceding intermediate. |
| P12 | The paper’s explicit longer route is c → e → ī → d̄ → j → i → ē → c̄. | Visually checked p.6670; bars are lost in plain OCR. | Retain the paper’s notation in a historical panel only. Our computational sign convention is not fully calibrated to the drawings. |
| P13 | The ideal collective flip energies increase from 1.6 to 6.9 to 10 kcal/mol for systems 4, 5, and 6. | p.6669. | This is a comparison within the paper’s MM2 construction, not an established scaling law for arbitrary chains or other force fields. |
| P14 | The paper’s lowest mechanisms for 5 and 6 are localized disrotations of neighboring ring pairs. | Abstract and pp.6668–6670. | A local pair motion is itself correlated; avoid treating the word correlated as synonymous with all rings moving together. |
| P15 | A flipping ring crosses a reference-plane normal; a nonflipping ring crosses the plane. | Introduction p.6662. | “One-ring flip” does not mean only one ring moves. Preserve winding and atom correspondence when testing this mechanism. |
| P16 | A driven terminal ring can produce a localized pathway after relaxation. | pp.6668–6669. | A drive or restraint supplies a proposal; the resulting mechanism must be measured. |
| P17 | Methyl conformations affect the calculated localized barriers. | pp.6668–6669 and footnote 22. | Permutation-equivalent methyl configurations can be duplicates, but not every methyl rearrangement is a symmetry-only self-loop. |

## Current MMFF94 numerical study

| ID | Current statement | Exact record / calculation | Boundary of the claim |
|---|---|---|---|
| N01 | The model is compound 6 with 99 explicit atoms, C43H56, and MMFF94 energies in kcal/mol. | `src/propeller_paths/data/molecule.json`; input hashes in every run report. | Graph, atom order, force-field variant, and nonbonded settings are packaged explicitly. |
| N02 | A and B are enantiomeric minima of equal energy. | Proper graph-only RMSD matching keeps them distinct; reflected structures and E/gradient parity are checked. | Reflection is a construction of a physical partner, not an allowed proper basin identification. |
| N03 | The direct A–C–M–B numerical route reaches 8.0411517 kcal/mol above A. | Baseline TS1 energy 212.552365828 and A energy 204.511214104. | State this as the earlier route, not the optimum after network expansion. |
| N04 | The best route in the final checked finite graph reaches 7.403994698690354 kcal/mol above A in the fresh standalone revalidation. | `results/summary.json`, `results/reference/network_full.json`, and `results/reference/standalone_sequential_summary.json`. | It is the graph minimax route with a revalidated finite path maximum, not proof of the globally lowest barrier on the continuous landscape. |
| N05 | The selected route is A–C–P–Q–D–M–B, with P=N002 and Q=N015 in the archived records. | Same final checked graph; bottleneck energy approximately 211.915208803 kcal/mol. | The route has six microscopic events, including a methyl event. It is not the same labeled sequence as the paper’s seven-edge example. |
| N06 | The selected localized route has six ordinary saddle connections, each with an index-one saddle and two index-zero endpoints, with proper symmetry matches. | Packaged edge/node-join records under `results/reference/sequential_edges/`; raw branch NPZ files are regenerated by the `sequential` command. | A high-order stationary point, an optimizer peak, or a nonstationary constrained maximum is not an ordinary accepted edge. |
| N07 | The recorded new paths passed dense finite energy checks, including connectors and joins. | `results/reference/standalone_sequential_summary.json` and per-pattern complete-path audits. | Finite probes are not a rigorous continuous supremum certificate or a mass-weighted IRC. |
| N08 | After reflection closure, 36 validated minima and 32 reachable minima represent all 20 coarse labels. | Final checked graph `coverage`. | Coarse coverage is constructive but does not imply every basin or transition was found. |
| N09 | The primary search used 32 scans with 608 scan configurations and retained failed proposals. | The `scans`, `proposals`, `expanded`, and `settings` records in `results/reference/network_full.json`. | Only four starting minima were expanded in all six local directions; finite budgets and targeted stops remain material. |
| N10 | Five events in the selected route predominantly involve an adjacent ring pair; the sixth is a methyl rearrangement. | `results/reference/mechanism_geometry.json`. | The measured ring-pair contributions are about 95–99%. A scan’s driver identity alone is insufficient. |
| N11 | Complete synchronized candidates I/III and II/IV reach 15.56590442550737 and 15.514201146247586 kcal/mol above A, respectively. | `results/summary.json` and the two passed independent complete-path audits under `results/reference/collective/`. | Converged branch nodes and explicit fixed-progress saddle connectors are included. Peripheral interpolation is checked but not fully minimized. These are candidate-path heights, not globally minimum collective barriers. |
| N12 | Potential-energy comparison does not determine experimental interconversion rates. | Definition of the model and measured quantity. | A thermal kinetic claim needs free-energy and rate analysis; the project has not performed those calculations. |
| N13 | The separate node study accepted 41 constrained minima for each pattern/direction combination: 164 node calculations in total. | Four branch reports under `results/reference/collective/node_studies/`, also summarized in `results/summary.json`. | A complete node sequence is not itself a checked continuous path. Its node maximum does not include every connecting segment's energy. |
| N14 | The user reported successful Mac Studio reproduction of both complete synchronized candidates, with four accepted constrained connectors and finite heights of 15.5659044 and 15.5142011 kcal/mol above A. | User-pasted stdout transcribed in `results/reported_mac/complete_path_summary.json`; evidence provenance and comparison in `results/reported_mac/provenance.json` and `results/reported_mac/comparison_with_local.json`. | The reported checks and heights agree with the locally audited reproduction. Raw Mac files and the executed source hash were not independently accessed; the II/IV frame count differs by two. This records a reported reproduction, not an independent raw-data audit or a global optimum. |

## Interpretation and publication wording

| Proposed wording | Status | Better wording |
|---|---|---|
| “We reproduced Lindner’s 1993 barriers.” | Unsupported | “We revisited the mechanistic question with MMFF94 and explicit path checks.” |
| “Our 7.4 kcal/mol barrier agrees with the experiment.” | Incorrect comparison | “The compound-6 MMFF94 route reaches 7.40399 kcal/mol above A. The historical 7.4 NMR upper bound concerned compound 5.” |
| “We found the minimum-energy path.” | Too broad | “We found the lowest bottleneck route in the finite checked graph.” |
| “All 20 states were found.” | Ambiguous | “The checked graph contains representatives of all 20 coarse propeller form/sign labels.” |
| “The entire molecule moves, so the mechanism is collective.” | Mechanistically inadequate | “The internal torsion traces identify which local rotations occur; global translation and bending are separate.” |
| “Identical endpoints imply the same mechanism.” | False | “Different angular windings can connect symmetry-equivalent endpoints.” |
| “The constrained peak is an ordinary molecular transition state.” | Unsupported | “The fixed-progress connector is an index-one saddle on the nonrigid constraint tangent space, not an unrestricted elementary transition state.” |
| “My supervisor’s first paper inspired this project.” | User-provided personal context | Acceptable in the user’s own draft, without claiming independent bibliographic verification or endorsement. |
| “Dr. Lindner collaborated on/endorses this repository.” | No evidence | Omit unless the user supplies explicit facts supporting that statement. |
| “We used a learned manifold/diffusion model.” | Unsupported for these runs | “These checked force-field paths can serve as a future benchmark for learned geometric proposals.” |

## Sources to retain

- Official ACS metadata: https://pubs.acs.org/toc/joceah/58/24 and https://doi.org/10.1021/jo00076a027.
- Locally supplied article PDF, read with printed page references and visual checks for overbars. Keep it outside a public source release by default.
- Immutable checked graph, stationary-point records, paths, manifests, and failed-proposal log from the present calculations.
- Final collective study, sensitivity records, exact source-bound independent audits, and the separate fresh standalone reproduction records under `results/reference/collective/`.
