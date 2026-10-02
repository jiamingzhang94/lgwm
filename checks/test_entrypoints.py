"""Checks for the command-line entry points and Stage A training."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from lgwm.train.stage_a import StageAModel,sample_block_mask
ROOT=Path(__file__).resolve().parents[1]

def tool(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class EntryTests(unittest.TestCase):
 def test_zero_coordinates_are_present(self):
  action=tool('score_transition').action_inputs({'action_type':'tap','x':0.,'y':0.},'cpu')
  self.assertTrue(action.a_has_coord.item());self.assertFalse(action.a_has_text.item())
 def test_partial_or_invalid_coordinates_fail(self):
  f=tool('score_transition').action_inputs
  for data in [{'action_type':'tap','x':.5},{'action_type':'tap','x':float('nan'),'y':.5},{'action_type':'scroll','direction':'diagonal'}]:
   with self.assertRaises(ValueError):f(data,'cpu')
 def test_nonempty_text_cannot_silently_become_zero(self):
  with self.assertRaises(ValueError):tool('score_transition').action_inputs({'action_type':'type_text','text':'hello'},'cpu')
 def test_block_mask_is_deterministic_and_nontrivial(self):
  for seed in range(20):
   mask=sample_block_mask(np.random.default_rng(seed))
   np.testing.assert_array_equal(mask,sample_block_mask(np.random.default_rng(seed)))
   self.assertEqual(mask.shape,(512,));self.assertTrue(mask.any());self.assertFalse(mask.all())
 @unittest.skipUnless(torch.cuda.is_available(),'GPU needed for training step')
 def test_stage_a_trains_context_and_keeps_target_frozen(self):
  model=StageAModel('vit_small_patch14_dinov2.lvd142m',pred_depth=1,pretrained=False).cuda()
  mask=torch.from_numpy(sample_block_mask(np.random.default_rng(7))).cuda().unsqueeze(0)
  image=torch.randint(0,256,(1,3,512,256),device='cuda',dtype=torch.uint8)
  loss=model(image,mask);loss.backward()
  self.assertTrue(torch.isfinite(loss))
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.vit.parameters()))
  self.assertTrue(all(p.grad is None for p in model.target.parameters()))

if __name__=='__main__':unittest.main()
