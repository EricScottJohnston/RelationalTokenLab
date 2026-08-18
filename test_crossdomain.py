import random, unittest
from crossdomain_data import *
from crossdomain_models import RelationalCaseModel
class Tests(unittest.TestCase):
    def test_toddler(self):
        r=random.Random(1); self.assertEqual(toddler_story(r,owner='A',b_authorized=False).label,ADVERSE); self.assertEqual(toddler_story(r,owner='B',a_permission='none').label,NON_ADVERSE); self.assertEqual(toddler_story(r,owner='B',a_permission='active').label,ADVERSE)
    def test_legal(self):
        r=random.Random(2); self.assertEqual(legal_story(r,plaintiff_right='yes',authority='invalid',serious=True).label,ADVERSE); self.assertEqual(legal_story(r,plaintiff_right='yes',authority='valid',serious=True).label,NON_ADVERSE); self.assertEqual(legal_story(r,plaintiff_right='no',authority='invalid',serious=True).label,NON_ADVERSE)
    def test_pairs(self):
        for a,b in legal_material_pairs(3,10): self.assertNotEqual(a.label,b.label)
        for a,b in legal_revelation_pairs(4,10): self.assertEqual(a.label,UNRESOLVED); self.assertNotEqual(b.label,UNRESOLVED)
    def test_freeze(self):
        m=RelationalCaseModel(100).transfer_shell(120); self.assertTrue(all(not p.requires_grad for p in m.operator.parameters())); self.assertFalse(m.initial_state.requires_grad); self.assertTrue(any(p.requires_grad for p in m.encoder.parameters()))
if __name__=='__main__': unittest.main()
