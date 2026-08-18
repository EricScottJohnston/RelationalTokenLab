import unittest

from relational_core import (
    RelationalGraph,
    compose_relations,
    complex_compose,
    loop_is_consistent,
    generate_consistent_graph,
)


class RelationalCoreTests(unittest.TestCase):
    def test_i_times_i_is_minus_one(self):
        self.assertEqual(compose_relations([1, 1]), 2)
        self.assertEqual(complex_compose([1, 1]), -1 + 0j)

    def test_consistent_loop(self):
        self.assertTrue(loop_is_consistent([1, 1, 2]))
        self.assertFalse(loop_is_consistent([1, 1, 1]))

    def test_graph_inference(self):
        g = RelationalGraph(3)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, 1)
        self.assertEqual(g.infer(0, 2), 2)
        ok, _ = g.check_consistency()
        self.assertTrue(ok)

    def test_graph_contradiction(self):
        g = RelationalGraph(3)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, 1)
        g.add_edge(0, 2, 1)  # should be 2
        ok, _ = g.check_consistency()
        self.assertFalse(ok)

    def test_generated_graph(self):
        g, hidden = generate_consistent_graph(seed=9)
        ok, reason = g.check_consistency()
        self.assertTrue(ok, reason)
        for a in range(g.node_count):
            for b in range(g.node_count):
                inferred = g.infer(a, b)
                expected = (hidden[b] - hidden[a]) % 4
                self.assertEqual(inferred, expected)


if __name__ == "__main__":
    unittest.main()
