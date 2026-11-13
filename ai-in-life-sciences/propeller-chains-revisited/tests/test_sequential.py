"""Small numerical tests for the two path failures encountered in this study."""
import unittest

import numpy as np

from propeller_paths.sequential import audit_trace, capped_minimize


class SequentialNumericalTests(unittest.TestCase):
    def test_gap_audit_rejects_high_energy_connector_and_preserves_markers(self):
        # Both endpoint energies are zero, while the connector crosses a barrier.
        def eg(x):
            energy = (1.0 - x[0] ** 2) ** 2
            gradient = np.zeros(297)
            gradient[0] = -4.0 * x[0] * (1.0 - x[0] ** 2)
            return energy, gradient

        points = np.zeros((2, 99, 3))
        points[0, 0, 0], points[1, 0, 0] = -1.0, 1.0
        xyz, energy, report = audit_trace({"eg": eg}, points, 0.5, anchor_indices=[0, 1])
        self.assertEqual(report["status"], "CONTINUITY_UNRESOLVED")
        self.assertGreater(report["maximum_sampled_energy_kcal_mol"], 0.999)
        self.assertLessEqual(report["maximum_checked_interval_atom_displacement_A"], 0.02000000001)
        self.assertTrue(np.array_equal(xyz[report["anchor_image_indices"]["0"]], points[0]))
        self.assertTrue(np.array_equal(xyz[report["anchor_image_indices"]["1"]], points[1]))
        self.assertEqual(float(energy[0]), 0.0)
        self.assertEqual(float(energy[-1]), 0.0)

    def test_capped_minimizer_records_every_accepted_bounded_step(self):
        target = np.linspace(-0.2, 0.2, 297)
        stiffness = np.linspace(0.5, 3.0, 297)

        def eg(x):
            displacement = x - target
            return float(np.sum(stiffness * displacement ** 2) / 2), stiffness * displacement

        initial = np.full(297, 0.35)
        recorded = [initial.copy()]
        final, report = capped_minimize(eg, initial, lambda x: recorded.append(x.copy()),
                                        iterations=1000)
        values = np.asarray(recorded)
        energies = np.asarray([eg(x)[0] for x in values])
        gaps = np.linalg.norm(np.diff(values, axis=0).reshape(-1, 99, 3), axis=2)
        self.assertEqual(report["status"], "GRADIENT_CONVERGED")
        self.assertEqual(len(recorded), report["accepted_steps"] + 1)
        self.assertTrue(np.array_equal(recorded[-1], final))
        self.assertLessEqual(float(gaps.max()), 0.02000000001)
        self.assertLessEqual(float(np.diff(energies).max()), 1e-12)
        self.assertLess(float(np.linalg.norm(eg(final)[1].reshape(99, 3), axis=1).max()), 1e-5)
