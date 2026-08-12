import unittest

import numpy as np

from src.runtime_intervention import conservative_empirical_threshold


class ThresholdCalibrationTests(unittest.TestCase):
    def test_conservative_threshold_respects_training_budget_with_inclusive_rule(self):
        values = np.arange(20, dtype=float)
        result = conservative_empirical_threshold(values, 0.05)
        self.assertLessEqual(np.mean(values >= result["threshold"]), 0.05)
        self.assertGreater(result["threshold"], result["order_statistic"])

    def test_tied_maxima_do_not_exceed_budget(self):
        values = np.array([0.0] * 18 + [1.0, 1.0])
        result = conservative_empirical_threshold(values, 0.05)
        self.assertEqual(np.mean(values >= result["threshold"]), 0.0)
        self.assertTrue(result["training_budget_respected"])


if __name__ == "__main__":
    unittest.main()
