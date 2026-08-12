import unittest

import numpy as np

from src.temporal_detectability import (
    build_public_discussion_index,
    build_temporal_representation,
    source_rounds,
)


class PrefixFeatureTests(unittest.TestCase):
    def setUp(self):
        self.vectors = {
            1: np.array([1.0, 10.0]),
            2: np.array([3.0, 30.0]),
            3: np.array([8.0, 80.0]),
            4: np.array([10000.0, 10000.0]),
        }

    def test_round_one_contains_only_round_one(self):
        np.testing.assert_array_equal(
            build_temporal_representation(self.vectors, "round_only", 1),
            self.vectors[1],
        )
        self.assertEqual(source_rounds("round_only", 1), (1,))

    def test_prefix_two_contains_only_rounds_one_and_two(self):
        expected = np.mean([self.vectors[1], self.vectors[2]], axis=0)
        np.testing.assert_array_equal(
            build_temporal_representation(self.vectors, "prefix", 2), expected
        )
        self.assertEqual(source_rounds("prefix", 2), (1, 2))

    def test_prefix_three_contains_only_rounds_one_two_and_three(self):
        expected = np.mean([self.vectors[1], self.vectors[2], self.vectors[3]], axis=0)
        np.testing.assert_array_equal(
            build_temporal_representation(self.vectors, "prefix", 3), expected
        )
        self.assertEqual(source_rounds("prefix", 3), (1, 2, 3))

    def test_final_round_is_excluded_from_index_and_all_pre_final_features(self):
        metadata = []
        activations = []
        for agent_number in range(4):
            for round_number in (1, 2, 3, 4):
                metadata.append({
                    "run_id": "deception__audit_01__control",
                    "scenario_id": "audit_01",
                    "scenario_domain": "audit",
                    "mode": "control",
                    "agent_name": "agent{}".format(agent_number),
                    "role": "honest",
                    "round": round_number,
                    "phase": "discussion" if round_number < 4 else "final",
                    "channel": "public",
                })
                activations.append(np.array([float(round_number)]))
        index, _ = build_public_discussion_index(metadata, np.array(activations))
        for rounds in index["deception__audit_01__control"].values():
            self.assertEqual(set(rounds), {1, 2, 3})
            self.assertNotIn(4, rounds)
        self.assertNotIn(4, source_rounds("prefix", 3))


if __name__ == "__main__":
    unittest.main()
