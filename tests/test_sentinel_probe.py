import unittest

import numpy as np
import pandas as pd

from src.sentinel_probe import (
    attach_paired_diagnostics,
    calculate_sentinel_metrics,
    filter_observer_examples,
    fit_sentinel_fold,
    run_lodo_fixed_sentinel,
)


def make_examples():
    rows = []
    for domain, offset in (("train", 0.0), ("test", 1000.0)):
        for scenario_number in range(2):
            scenario = "{}_{}".format(domain, scenario_number)
            for position in (2, 3):
                style_matched = not (scenario_number == 1 and position == 3)
                for mode, label, signal in (
                    ("control", 0, -1.0),
                    ("collusion", 1, 1.0),
                ):
                    rows.append({
                        "scenario_id": scenario,
                        "run_id": "{}__{}".format(scenario, mode),
                        "domain": domain,
                        "mode": mode,
                        "environment_label": label,
                        "agent_name": "{}_p{}".format(scenario, position),
                        "role": "honest",
                        "generator_position": position,
                        "paired_prompt_style_matched": style_matched,
                        "temporal_view": "round_only",
                        "round": 1,
                        "round_or_prefix": "R1",
                        "source_rounds": "1",
                        "activation": np.array(
                            [signal + offset, 2.0 * signal + offset], dtype=float
                        ),
                    })
    return pd.DataFrame(rows)


class SentinelProbeTests(unittest.TestCase):
    def test_fixed_full_discussion_probe_is_reused_at_every_temporal_point(self):
        base = make_examples()
        rows = []
        for _, row in base.iterrows():
            for temporal_view in ("round_only", "prefix"):
                for round_number in (1, 2, 3):
                    value = row.to_dict()
                    value["temporal_view"] = temporal_view
                    value["round"] = round_number
                    value["round_or_prefix"] = (
                        "R{}".format(round_number)
                        if temporal_view == "round_only"
                        else "-".join("R{}".format(number) for number in range(1, round_number + 1))
                    )
                    value["source_rounds"] = (
                        str(round_number)
                        if temporal_view == "round_only"
                        else "|".join(str(number) for number in range(1, round_number + 1))
                    )
                    value["activation"] = np.asarray(row["activation"]) + round_number * 0.1
                    rows.append(value)
        examples = pd.DataFrame(rows)
        predictions, _, audits = run_lodo_fixed_sentinel(
            examples, ["train", "test"]
        )
        fingerprint_counts = predictions.groupby(
            ["analysis_population", "observer_scope", "fold"]
        )["model_fingerprint"].nunique()
        self.assertTrue((fingerprint_counts == 1).all())
        self.assertTrue(predictions["fixed_across_temporal_views"].all())
        self.assertEqual(set(predictions["probe_training_representation"]), {"mean(R1,R2,R3)"})
        fitted = [audit for audit in audits if audit["fit_status"] == "fit"]
        self.assertTrue(fitted)
        self.assertTrue(all(audit["training_source_rounds"] == "1|2|3" for audit in fitted))
        self.assertTrue(all(not audit["held_out_domain_present_in_training"] for audit in fitted))

    def test_style_filter_reports_only_exact_matched_pairs(self):
        examples = make_examples()
        selected = filter_observer_examples(examples, "style_matched", "both_positions")
        self.assertTrue(selected["paired_prompt_style_matched"].all())
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected[["scenario_id", "agent_name"]].drop_duplicates().shape[0], 6)

    def test_scaler_and_probe_use_no_held_out_domain(self):
        examples = make_examples()
        train = examples.loc[examples["domain"] == "train"]
        test = examples.loc[examples["domain"] == "test"]
        _, _, audit = fit_sentinel_fold(train, test, "test")
        expected = np.mean(np.stack(train["activation"]), axis=0)
        self.assertEqual(audit["training_domains"], ["train"])
        self.assertFalse(audit["held_out_domain_present_in_training"])
        # A giant held-out offset would dominate this mean if leakage occurred.
        np.testing.assert_allclose(expected, np.array([0.0, 0.0]))
        self.assertEqual(audit["scaler"]["n_samples_seen"], len(train))

    def test_paired_projection_matches_decision_score_shift(self):
        examples = make_examples()
        train = examples.loc[examples["domain"] == "train"]
        test = examples.loc[examples["domain"] == "test"]
        prediction_records, diagnostic_records, _ = fit_sentinel_fold(train, test, "test")
        predictions = pd.DataFrame(prediction_records)
        predictions["analysis_population"] = "all_observers"
        predictions["observer_scope"] = "both_positions"
        diagnostics = pd.DataFrame(diagnostic_records)
        diagnostics["analysis_population"] = "all_observers"
        diagnostics["observer_scope"] = "both_positions"
        diagnostics["temporal_view"] = "round_only"
        diagnostics["round"] = 1
        diagnostics["round_or_prefix"] = "R1"
        diagnostics["source_rounds"] = "1"
        attached = attach_paired_diagnostics(predictions, diagnostics)
        np.testing.assert_allclose(
            attached["paired_score_difference"], attached["sentinel_decision_score_shift"]
        )

    def test_missing_style_matched_fold_is_explicit(self):
        examples = make_examples()
        # Retain paired rows only from the train domain, leaving test undefined.
        predictions = []
        for domain in ("train",):
            frame = examples.loc[(examples["domain"] == domain) & (examples["generator_position"] == 2)]
            for _, row in frame.iterrows():
                predictions.append({
                    "analysis_population": "style_matched",
                    "observer_scope": "position_2",
                    "temporal_view": "round_only",
                    "round": 1,
                    "round_or_prefix": "R1",
                    "scenario_id": row["scenario_id"],
                    "agent_name": row["agent_name"],
                    "generator_position": 2,
                    "fold": domain,
                    "environment_label": row["environment_label"],
                    "sentinel_score": float(row["environment_label"]),
                    "paired_score_difference": 1.0,
                })
        folds, summary, issues = calculate_sentinel_metrics(
            pd.DataFrame(predictions), ["train", "test"], bootstrap_replicates=10
        )
        missing = folds.loc[folds["fold"] == "test"].iloc[0]
        self.assertTrue(pd.isna(missing["auroc"]))
        self.assertEqual(missing["metric_issue"], "no retained observer pairs in held-out domain")
        self.assertEqual(summary.iloc[0]["fold_count_undefined"], 1)
        self.assertEqual(len(issues), 1)


if __name__ == "__main__":
    unittest.main()
