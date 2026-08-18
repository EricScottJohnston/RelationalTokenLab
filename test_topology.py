import random
import unittest

from topology_models import (
    UNKNOWN,
    check_coherence_exact,
    infer_relation_exact,
    make_dynamic_episode,
)


class TopologyTests(unittest.TestCase):
    def test_dynamic_episode_semantics(self):
        rng = random.Random(42)
        episode = make_dynamic_episode(rng, 16)
        self.assertEqual([name for name, _ in episode],
                         ["BASE", "CUT", "RECONNECT", "CONTRADICTION", "REPAIR"])

        base = episode[0][1]
        cut = episode[1][1]
        reconnect = episode[2][1]
        contradiction = episode[3][1]
        repair = episode[4][1]

        self.assertTrue(check_coherence_exact(base.node_count, base.edges))
        self.assertNotEqual(
            infer_relation_exact(base.node_count, base.edges, base.src, base.dst),
            UNKNOWN,
        )

        self.assertTrue(check_coherence_exact(cut.node_count, cut.edges))
        self.assertEqual(
            infer_relation_exact(cut.node_count, cut.edges, cut.src, cut.dst),
            UNKNOWN,
        )

        self.assertTrue(check_coherence_exact(reconnect.node_count, reconnect.edges))
        self.assertEqual(
            infer_relation_exact(
                reconnect.node_count, reconnect.edges, reconnect.src, reconnect.dst
            ),
            reconnect.relation_target,
        )

        self.assertFalse(
            check_coherence_exact(contradiction.node_count, contradiction.edges)
        )

        self.assertTrue(check_coherence_exact(repair.node_count, repair.edges))


if __name__ == "__main__":
    unittest.main()
