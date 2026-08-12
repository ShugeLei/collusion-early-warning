import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "narcbench"
sys.path.insert(0, str(UPSTREAM_ROOT))

from probes import reproduce as upstream
from src.temporal_detectability import (
    aggregate_run_scores,
    official_probe_candidate_indices,
    score_fold_agents,
    validate_oof_predictions,
)


class FakeArchive:
    def __init__(self, array):
        self.array = array
        self.files = ["layer_28"]

    def __getitem__(self, key):
        if key != "layer_28":
            raise KeyError(key)
        return self.array


def synthetic_probe_data(test_offset):
    rows = []
    vectors = []
    for domain, scenario, offset in (
        ("train", "train_01", 0.0),
        ("test", "test_01", test_offset),
    ):
        for role, base in (("honest", -1.0), ("colluder", 1.0)):
            for repetition in range(3):
                rows.append({
                    "run_id": "deception__{}__collusion".format(scenario),
                    "scenario_id": scenario,
                    "scenario_domain": domain,
                    "mode": "collusion",
                    "channel": "public",
                    "phase": "discussion",
                    "round": repetition + 1,
                    "agent_name": "{}_{}".format(role, repetition),
                    "role": role,
                })
                vectors.append(np.array([base + offset, base * 2 + offset], dtype=np.float32))
    return rows, np.array(vectors)


class LeakageTests(unittest.TestCase):
    def test_held_out_rows_are_not_probe_candidates(self):
        metadata, _ = synthetic_probe_data(1000.0)
        audit = official_probe_candidate_indices(
            metadata, "test", upstream._domain_from_scenario_id
        )
        selected = audit["selected_honest"] + audit["selected_colluder"]
        self.assertTrue(selected)
        self.assertTrue(all(metadata[index]["scenario_domain"] == "train" for index in selected))

    def test_changing_held_out_activations_changes_neither_scaler_nor_direction(self):
        metadata_a, vectors_a = synthetic_probe_data(1000.0)
        metadata_b, vectors_b = synthetic_probe_data(-1000.0)
        scaler_a, direction_a = upstream.train_deception_probe(
            metadata_a, FakeArchive(vectors_a), 28, exclude_domain="test"
        )
        scaler_b, direction_b = upstream.train_deception_probe(
            metadata_b, FakeArchive(vectors_b), 28, exclude_domain="test"
        )
        np.testing.assert_array_equal(scaler_a.mean_, scaler_b.mean_)
        np.testing.assert_array_equal(scaler_a.scale_, scaler_b.scale_)
        np.testing.assert_array_equal(direction_a, direction_b)

    def test_one_probe_fingerprint_is_reused_across_all_temporal_views(self):
        metadata, vectors = synthetic_probe_data(10.0)
        scaler, direction = upstream.train_deception_probe(
            metadata, FakeArchive(vectors), 28, exclude_domain="test"
        )
        temporal_index = {}
        run_metadata = {}
        for mode in ("collusion", "control"):
            run_id = "deception__test_01__{}".format(mode)
            temporal_index[run_id] = {
                "agent{}".format(agent): {
                    round_number: np.array([agent + round_number, agent - round_number], dtype=float)
                    for round_number in (1, 2, 3)
                }
                for agent in range(4)
            }
            run_metadata[run_id] = {
                "scenario_id": "test_01",
                "domain": "test",
                "mode": mode,
                "roles": {"agent{}".format(agent): "honest" for agent in range(4)},
            }
        records = score_fold_agents(
            temporal_index, run_metadata, "test", scaler, direction, 28
        )
        frame = pd.DataFrame(records)
        self.assertEqual(frame["probe_fingerprint"].nunique(), 1)
        self.assertEqual(set(frame["temporal_view"]), {"round_only", "prefix"})
        self.assertEqual(set(frame["round"]), {1, 2, 3})
        run_frame = aggregate_run_scores(frame)
        self.assertEqual(validate_oof_predictions(frame, run_frame, ["test"]), [])


if __name__ == "__main__":
    unittest.main()
