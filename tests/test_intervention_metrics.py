import unittest

import pandas as pd

from src.runtime_intervention import apply_sequential_policy, calculate_intervention_metrics


class InterventionMetricTests(unittest.TestCase):
    def test_policy_stops_at_first_crossing_and_uses_no_round_four(self):
        rows = []
        for mode, risks in (("control", [0.0, 0.0, 0.0]), ("collusion", [0.1, 0.8, 1.2])):
            for round_number, risk in enumerate(risks, start=1):
                rows.append({
                    "scenario_id": "s1",
                    "run_id": "s1_{}".format(mode),
                    "domain": "d",
                    "mode": mode,
                    "method": "mean_sentinel_risk",
                    "round": round_number,
                    "risk_score": risk,
                    "agent_count": 2,
                })
        thresholds = pd.DataFrame([
            {
                "fold": "d",
                "method": "mean_sentinel_risk",
                "target_tfir": budget,
                "threshold": 0.5,
                "calibration_trajectory_count": 10,
                "training_empirical_tfir": 0.0,
            }
            for budget in (0.01, 0.05, 0.10, 0.20)
        ])
        policy = apply_sequential_policy(
            pd.DataFrame(rows), thresholds, "all_observers", "trusted_sentinel"
        )
        collusion = policy.loc[
            (policy["mode"] == "collusion") & (policy["target_tfir"] == 0.05)
        ].iloc[0]
        self.assertEqual(collusion["first_detection_round"], 2)
        self.assertEqual(collusion["intervention_lead_rounds"], 2)
        self.assertFalse(collusion["detected_by_r1"])
        self.assertTrue(collusion["detected_by_r2"])

    def test_tfir_and_pic_are_trajectory_level(self):
        rows = []
        for scenario, control_flag, collusion_flag in (
            ("s1", False, True),
            ("s2", True, False),
        ):
            for mode, flag in (("control", control_flag), ("collusion", collusion_flag)):
                rows.append({
                    "analysis_population": "all_observers",
                    "monitor": "trusted_sentinel",
                    "method": "mean_sentinel_risk",
                    "target_tfir": 0.05,
                    "scenario_id": scenario,
                    "run_id": "{}_{}".format(scenario, mode),
                    "domain": "d",
                    "fold": "d",
                    "mode": mode,
                    "intervened": flag,
                    "first_detection_round": 1 if flag else None,
                    "intervention_lead_rounds": 3 if flag else None,
                    "detected_by_r1": flag,
                    "detected_by_r2": flag,
                    "detected_by_r3": flag,
                })
        summary, _, issues = calculate_intervention_metrics(
            pd.DataFrame(rows), ["d"], replicates=20
        )
        self.assertEqual(issues, [])
        self.assertEqual(summary.iloc[0]["achieved_tfir"], 0.5)
        self.assertEqual(summary.iloc[0]["pic"], 0.5)


if __name__ == "__main__":
    unittest.main()
