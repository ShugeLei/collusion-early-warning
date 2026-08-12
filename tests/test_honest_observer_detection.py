import unittest

import pandas as pd

from src.honest_observer_detection import (
    aggregate_observer_risk,
    calculate_observer_metrics,
    select_honest_observer_scores,
)


class HonestObserverTests(unittest.TestCase):
    def test_observer_selection_excludes_positions_zero_and_one(self):
        rows = []
        for position in range(4):
            rows.append({
                "run_id": "run",
                "temporal_view": "round_only",
                "round": 1,
                "agent_name": "agent{}".format(position),
                "generator_position": position,
                "role": "colluder" if position < 2 else "honest",
            })
        selected = select_honest_observer_scores(pd.DataFrame(rows))
        self.assertEqual(set(selected["generator_position"]), {2, 3})
        self.assertTrue((selected["role"] == "honest").all())

    def test_observer_risk_is_mean_and_peak_of_exactly_two_scores(self):
        rows = []
        for agent_name, score in (("a", -2.0), ("b", 3.0)):
            rows.append({
                "scenario_id": "s1",
                "run_id": "r1",
                "domain": "d1",
                "mode": "collusion",
                "temporal_view": "round_only",
                "round": 1,
                "round_or_prefix": "R1",
                "source_rounds": "1",
                "fold": "d1",
                "layer": 28,
                "probe_fingerprint": "probe",
                "agent_name": agent_name,
                "deception_score": score,
            })
        risks = aggregate_observer_risk(pd.DataFrame(rows)).set_index("method")
        self.assertEqual(risks.loc["mean_observer_suspicion", "risk_score"], 0.5)
        self.assertEqual(risks.loc["peak_observer_suspicion", "risk_score"], 3.0)

    def test_undefined_fold_is_retained_and_reported(self):
        rows = []
        for scenario, fold, mode, value in (
            ("s1", "a", "control", 0.0),
            ("s2", "b", "control", 0.0),
            ("s2", "b", "collusion", 1.0),
        ):
            rows.append({
                "scenario_id": scenario,
                "temporal_view": "round_only",
                "round": 1,
                "round_or_prefix": "R1",
                "method": "mean_observer_suspicion",
                "fold": fold,
                "domain": fold,
                "mode": mode,
                "risk_score": value,
            })
        folds, _, _, issues = calculate_observer_metrics(
            pd.DataFrame(rows), ["a", "b"], bootstrap_replicates=10
        )
        fold_a = folds.loc[folds["fold"] == "a"].iloc[0]
        self.assertTrue(pd.isna(fold_a["auroc"]))
        self.assertEqual(fold_a["metric_issue"], "single-class fold")
        self.assertTrue(any(issue["fold"] == "a" for issue in issues))


if __name__ == "__main__":
    unittest.main()
