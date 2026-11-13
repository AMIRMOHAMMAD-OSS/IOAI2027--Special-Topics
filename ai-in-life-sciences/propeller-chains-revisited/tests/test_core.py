"""Numerical contracts that can invalidate a scientific result."""
import unittest
import numpy as np
from propeller_paths.core import load_context, verify_core, maxg
from propeller_paths.constraints import check_torsion_jacobian, torsions_jac, wrapped
from propeller_paths.collective import endpoint_targets


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = load_context()

    def test_force_cache_and_proper_enantiomer_matching(self):
        result = verify_core(self.ctx, stationary=False)
        self.assertGreater(result["proper_A_B_rmsd_A"], 1)
        self.assertLess(result["reflected_A_B_rmsd_A"], 1e-8)
        self.assertLess(max(result["directional_gradient_errors"]), 1e-5)

    def test_analytic_torsions_against_independent_differences(self):
        probe = self.ctx["points"]["A"] + np.random.default_rng(1993).normal(0, .002, 297)
        result = check_torsion_jacobian(probe)
        self.assertLess(result["maximum_absolute_error_rad_per_A"], 1e-7)

    def test_collective_endpoint_windings_keep_graph_and_energy(self):
        for pattern in ("I_III", "II_IV"):
            a_xyz, b_xyz, a, delta = endpoint_targets(self.ctx, pattern)
            self.assertLess(abs(self.ctx["eg"](a_xyz)[0] - self.ctx["eg"](b_xyz)[0]), 1e-8)
            self.assertLess(maxg(self.ctx["eg"](b_xyz)[1]), 1e-5)
            self.assertLess(np.max(np.abs(wrapped(torsions_jac(b_xyz)[0] - a - delta))), 1e-12)


if __name__ == "__main__":
    unittest.main()
