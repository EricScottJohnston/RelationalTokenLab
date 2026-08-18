import random
import unittest

from system_grammar_data import (
    ROLES, OBSERVATION, INTERVENTION,
    generate_dataset, make_case, tg_class,
    TOPOLOGY_ONLY, GEOMETRY_ONLY,
)
from system_grammar_models import SystemGrammarModel


class Tests(unittest.TestCase):
    def test_policy_topology_vs_geometry(self):
        rng = random.Random(1)
        seen_t = False
        seen_g = False
        for _ in range(200):
            c = make_case(rng, "administrative", force_policy_structural=True, observation_probability=0.0)
            if c.topology_geometry == TOPOLOGY_ONLY:
                self.assertEqual(c.delta[ROLES.index("P")], 1)
                self.assertEqual(c.delta[ROLES.index("T")], 1)
                self.assertEqual(c.delta[ROLES.index("G")], 0)
                seen_t = True
            if c.topology_geometry == GEOMETRY_ONLY:
                self.assertEqual(c.delta[ROLES.index("P")], 1)
                self.assertEqual(c.delta[ROLES.index("G")], 1)
                self.assertEqual(c.delta[ROLES.index("T")], 0)
                seen_g = True
        self.assertTrue(seen_t and seen_g)

    def test_observation_is_not_intervention(self):
        rng = random.Random(2)
        c = make_case(rng, "mechanical", observation_probability=1.0)
        self.assertEqual(c.action_kind, OBSERVATION)
        self.assertEqual(sum(c.delta), 0)

    def test_compound_has_multiple_roles(self):
        data = generate_dataset(3, 50, "administrative", compound_sizes=(3,), observation_probability=0.0)
        self.assertTrue(any(sum(c.delta) >= 2 for c in data))

    def test_transfer_freezes_core_and_heads(self):
        base = SystemGrammarModel(100)
        shell = base.transfer_shell(120)
        self.assertTrue(all(not p.requires_grad for p in shell.core.parameters()))
        self.assertTrue(all(not p.requires_grad for p in shell.heads.parameters()))
        self.assertTrue(any(p.requires_grad for p in shell.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
