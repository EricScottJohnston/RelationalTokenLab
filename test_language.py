import random
import unittest
import torch

from language_models import (
    WORD_TO_ID,
    MAX_PHRASE_TOKENS,
    make_language_batch,
    LanguagePhaseModel,
)


class LanguageExperimentTests(unittest.TestCase):
    def test_batch_targets_are_mod4(self):
        rng = random.Random(1)
        batch = make_language_batch(
            batch_size=20,
            min_len=1,
            max_len=5,
            pad_edges_to=8,
            device=torch.device("cpu"),
            rng=rng,
        )
        self.assertEqual(batch.targets.shape[0], 20)
        self.assertTrue(bool(((batch.targets >= 0) & (batch.targets <= 3)).all()))

    def test_phase_model_shapes(self):
        rng = random.Random(2)
        batch = make_language_batch(
            batch_size=4,
            min_len=1,
            max_len=3,
            pad_edges_to=6,
            device=torch.device("cpu"),
            rng=rng,
        )
        model = LanguagePhaseModel(len(WORD_TO_ID))
        logits = model(batch.phrase_tokens, batch.edge_mask)
        self.assertEqual(tuple(logits.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
