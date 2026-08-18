import random
import unittest
import torch

from operator_models import (
    LearnedBinaryOperator,
    ground_truth_compose,
    make_dynamic_episode,
    exact_detect_coherence,
    exact_infer_state,
    UNKNOWN,
)


class OperatorExperimentTests(unittest.TestCase):
    def test_hidden_world_composition(self):
        self.assertEqual(ground_truth_compose([1, 1]), 2)
        self.assertEqual(ground_truth_compose([1, 1, 1, 1]), 0)
        self.assertEqual(ground_truth_compose([3, 1]), 0)

    def test_learned_operator_has_no_fixed_table(self):
        model = LearnedBinaryOperator(latent_dim=8, structured=True)
        self.assertTrue(hasattr(model, "operator"))
        self.assertEqual(sum(p.numel() for p in model.operator.parameters()) > 0, True)

    def test_dynamic_episode_ground_truth(self):
        ep = make_dynamic_episode(random.Random(7), 16)
        self.assertEqual(
            [x.name for x in ep],
            ["BASE", "CUT", "RECONNECT", "CONTRADICTION", "REPAIR"],
        )
        self.assertNotEqual(exact_infer_state(ep[0]), UNKNOWN)
        self.assertEqual(exact_infer_state(ep[1]), UNKNOWN)
        self.assertTrue(exact_detect_coherence(ep[0]))
        self.assertTrue(exact_detect_coherence(ep[1]))
        self.assertTrue(exact_detect_coherence(ep[2]))
        self.assertFalse(exact_detect_coherence(ep[3]))
        self.assertTrue(exact_detect_coherence(ep[4]))


if __name__ == "__main__":
    unittest.main()
