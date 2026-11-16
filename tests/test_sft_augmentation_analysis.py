"""Known paired statistical outcomes; no model, network or GPU."""
import unittest
from utils.report_sft_augmentation import paired,wilson,metrics
class PairedAnalysisTests(unittest.TestCase):
 def record(self,i,correct,length=10):
  return {'id':str(i),'correct':correct,'truncated':False,'full_trace_verified':False,'answer_extraction_success':True,'completion_token_ids':[1]*length}
 def test_exact_mcnemar_and_paired_length(self):
  a=[self.record(i,False,10) for i in range(8)]
  b=[self.record(i,True,7) for i in range(8)]
  r=paired(a,b)
  self.assertAlmostEqual(r['correct']['exact_mcnemar_p'],.0078125)
  self.assertEqual(r['correct']['mean_paired_difference'],1.)
  self.assertEqual(r['all_length']['paired_bootstrap95'],[-3.,-3.])
  self.assertEqual(r['common_correct_complete_length']['n'],0)
 def test_wilson_boundary_unknown_correct_length_and_pair_identity(self):
  lo,hi=wilson(0,50);self.assertAlmostEqual(lo,0);self.assertAlmostEqual(hi,.07134759913335872)
  r=metrics([self.record(1,False)]);self.assertIsNone(r['mean_correct_complete_tokens'])
  with self.assertRaises(ValueError):paired([self.record(1,True)],[self.record(2,True)])
if __name__=='__main__':unittest.main()
