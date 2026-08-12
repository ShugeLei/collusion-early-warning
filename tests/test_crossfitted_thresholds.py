import unittest

import numpy as np
import pandas as pd

from src.crossfitted_thresholds import (
    build_crossfitted_policy,
    empirical_tail_rank,
    experiment_e_domain_shift_audit,
    validate_crossfitted_predictions,
)


class CrossfittedThresholdTests(unittest.TestCase):
    def test_empirical_tail_rank_is_model_relative_and_tie_conservative(self):
        reference = np.array([1.0, 2.0, 2.0, 4.0])
        scores = np.array([0.0, 2.0, 3.0, 5.0])
        result = empirical_tail_rank(scores, reference)
        np.testing.assert_allclose(result, [0.0, 0.25, 0.75, 1.0])

    def test_domain_audit_does_not_claim_tiny_domain_quantiles(self):
        rows = []
        for mode, intervened, risk in (("control", True, 2.0), ("collusion", True, 3.0)):
            rows.append({
                "analysis_population": "all_observers",
                "monitor": "trusted_sentinel",
                "method": "mean_sentinel_risk",
                "fold": "d",
                "target_tfir": 0.05,
                "mode": mode,
                "threshold": 1.0,
                "calibration_trajectory_count": 20,
                "calibration_training_empirical_tfir": 0.05,
                "risk_r1": risk,
                "risk_r1_r2": risk - 1,
                "risk_r1_r2_r3": risk - 2,
                "intervened": intervened,
            })
        audit = experiment_e_domain_shift_audit(pd.DataFrame(rows), minimum_quantile_n=10)
        self.assertEqual(len(audit), 1)
        self.assertIsNone(audit[0]["benign_risk_p90"])
        self.assertIsNone(audit[0]["benign_risk_p95"])
        self.assertFalse(audit[0]["high_quantiles_reported"])

    def test_provenance_validator_detects_inner_training_leakage(self):
        rows = []
        for record_type, domain, inner_fold, target in (
            ("inner_calibration", "b", "b", np.nan),
            ("outer_test", "a", np.nan, 0.05),
        ):
            for round_number in (1, 2, 3):
                rows.append({
                    "record_type": record_type,
                    "outer_fold": "a",
                    "inner_fold": inner_fold,
                    "domain": domain,
                    "mode": "control",
                    "run_id": "{}_run".format(domain),
                    "round": round_number,
                    "normalized_tail_rank": 0.5,
                    "observer_positions": "2|3",
                    "observer_roles": "honest|honest",
                    "model_training_domains": "b|c" if record_type == "inner_calibration" else "b|c",
                    "target_tfir": target,
                })
        issues = validate_crossfitted_predictions(pd.DataFrame(rows), ["a"])
        self.assertIn("An inner fold entered model training", issues)

    def test_runtime_policy_thresholds_normalized_not_raw_risk(self):
        scores = pd.DataFrame([
            {
                "scenario_id": "s1",
                "run_id": "s1_control",
                "domain": "a",
                "mode": "control",
                "outer_fold": "a",
                "round": endpoint,
                "raw_risk": raw,
                "normalized_tail_rank": normalized,
            }
            for endpoint, raw, normalized in (
                (1, 100.0, 0.10),
                (2, 1.0, 0.60),
                (3, 0.0, 0.20),
            )
        ])
        thresholds = pd.DataFrame([
            {
                "outer_fold": "a",
                "target_tfir": target,
                "threshold": 0.50,
                "calibration_trajectory_count": 20,
                "training_empirical_tfir": 0.05,
            }
            for target in (0.01, 0.05, 0.10, 0.20)
        ])
        policy = build_crossfitted_policy(scores, thresholds)
        self.assertTrue(policy["intervened"].all())
        self.assertTrue((policy["first_detection_round"] == 2).all())


if __name__ == "__main__":
    unittest.main()
