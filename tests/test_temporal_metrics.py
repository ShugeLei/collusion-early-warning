import unittest

import pandas as pd

from src.temporal_detectability import calculate_auroc_metrics


class TemporalMetricTests(unittest.TestCase):
    def test_fold_and_pooled_auroc_are_computed_from_oof_runs(self):
        rows = []
        for fold in ("a", "b"):
            rows.extend([
                {
                    "temporal_view": "prefix",
                    "round": 1,
                    "round_or_prefix": "R1",
                    "method": "peak_suspicion",
                    "fold": fold,
                    "mode": "control",
                    "risk_score": 0.0,
                },
                {
                    "temporal_view": "prefix",
                    "round": 1,
                    "round_or_prefix": "R1",
                    "method": "peak_suspicion",
                    "fold": fold,
                    "mode": "collusion",
                    "risk_score": 1.0,
                },
            ])
        folds, summary, issues = calculate_auroc_metrics(pd.DataFrame(rows), ["a", "b"])
        self.assertEqual(issues, [])
        self.assertTrue((folds["auroc"] == 1.0).all())
        self.assertEqual(summary.iloc[0]["fold_macro_mean_auroc"], 1.0)
        self.assertEqual(summary.iloc[0]["fold_population_std_auroc"], 0.0)
        self.assertEqual(summary.iloc[0]["pooled_oof_auroc"], 1.0)

    def test_single_class_fold_is_reported_not_silently_dropped(self):
        frame = pd.DataFrame([
            {
                "temporal_view": "round_only",
                "round": 1,
                "round_or_prefix": "R1",
                "method": "deception_split",
                "fold": "a",
                "mode": "control",
                "risk_score": 0.0,
            },
            {
                "temporal_view": "round_only",
                "round": 1,
                "round_or_prefix": "R1",
                "method": "deception_split",
                "fold": "b",
                "mode": "control",
                "risk_score": 0.0,
            },
            {
                "temporal_view": "round_only",
                "round": 1,
                "round_or_prefix": "R1",
                "method": "deception_split",
                "fold": "b",
                "mode": "collusion",
                "risk_score": 1.0,
            },
        ])
        folds, _, issues = calculate_auroc_metrics(frame, ["a", "b"])
        fold_a = folds.loc[folds["fold"] == "a"].iloc[0]
        self.assertTrue(pd.isna(fold_a["auroc"]))
        self.assertEqual(fold_a["metric_issue"], "single-class fold")
        self.assertTrue(any(issue["fold"] == "a" for issue in issues))


if __name__ == "__main__":
    unittest.main()
