"""CPU-only tests for read-only SFT C diagnostics and actual serialization."""
import copy
import json
import math
import unittest
from pathlib import Path
import ast

from utils.run_dpo_sft_c_gravity_diagnosis import (
    ROOT, BASE, RUN, actual_teachers, sft_apis, serialized_audit, public_solver,
    nl_api, ols_certified, same_archived_audit)


class TeacherDiagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        cls.teachers, cls.original, cls.added = actual_teachers()
        cls.ns = nl_api()
        cls.solver = staticmethod(public_solver())

    def test_actual_64_plus_10_inclusion(self):
        self.assertEqual(sum(g == 'historical_r1' for g, *_ in self.teachers), 64)
        self.assertEqual(sum(g == 'added_real_c' for g, *_ in self.teachers), 10)

    def test_all_actual_teachers_independent_math(self):
        for group, record, source, api in self.teachers:
            ref = self.ns['gravity_nl_public_reference'](record['prompt'])
            sol = self.solver(record['prompt'])
            self.assertTrue(math.isclose(ref['g_ols'], sol['trace']['g_ols'], rel_tol=1e-12))
            self.assertEqual(record['final_answer'], ref['answer'])
            audit = self.ns['verify_gravity_natural_language'](record['prompt'], record['completion'], record['final_answer'])
            self.assertTrue(audit['verified'], (group, record['source_id'], audit['reasons']))

    def test_actual_mask_includes_ols_g_query_raw_box(self):
        for _, record, _, api in self.teachers:
            a = serialized_audit(self.tok, record, api)
            self.assertTrue(a['prompt_fully_masked'])
            self.assertTrue(a['response_fully_supervised'])
            self.assertTrue(a['all_required_steps_supervised'])
            self.assertTrue(a['no_truncation'])
            self.assertTrue(a['steps']['ols_formula']['inside_thinking'])
            self.assertFalse(a['steps']['boxed_answer']['inside_thinking'])

    def test_actual_target_final_channel_is_boxed_only(self):
        for _, record, _, api in self.teachers:
            a = serialized_audit(self.tok, record, api)
            self.assertEqual(a['actual_final_section'], 'Final answer: \\boxed{' + record['final_answer'] + '}')
            audit = self.ns['verify_gravity_natural_language'](record['prompt'], a['serialized_response'], record['final_answer'], thinking_expected=True)
            self.assertTrue(audit['final_correct'])
            self.assertFalse(audit['verified'])
            self.assertIn('all_example_ols_fit_not_demonstrated', audit['reasons'])

    def test_loss_causal_mask_and_gradient_match_dense(self):
        import torch
        from types import SimpleNamespace
        _, current = sft_apis()
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.randn(1, 10, 17))
            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.logits)
        torch.manual_seed(3)
        model = Tiny()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        labels = ids.clone()
        labels[:, :4] = -100
        batch = {'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids)}
        loss, count = current['course_sft_loss_sum'](model, batch, 3)
        gradient = torch.autograd.grad(loss, model.logits, retain_graph=True)[0]
        dense = torch.nn.functional.cross_entropy(model.logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1), reduction='sum', ignore_index=-100)
        expected = torch.autograd.grad(dense, model.logits)[0]
        self.assertEqual(count, 6)
        torch.testing.assert_close(loss, dense)
        torch.testing.assert_close(gradient, expected)
        self.assertEqual(float(gradient[:, :3].abs().sum()), 0.)
        self.assertTrue(all(float(gradient[:, i].abs().sum()) > 0 for i in range(3, 9)))

    def test_changed_g_and_raw_rejected_even_with_correct_final(self):
        _, record, _, _ = self.teachers[0]
        text = record['completion'].replace('The fitted g is ', 'The fitted g is 999; original value ')
        a = self.ns['verify_gravity_natural_language'](record['prompt'], text, record['final_answer'])
        self.assertFalse(a['verified'])

    def test_ols_keyword_alone_not_correct_ols(self):
        a = {'reference': {'g_ols': 10.}, 'reported_g_values': [99.], 'checks': {'all_public_ols': True}, 'reasons': []}
        self.assertFalse(ols_certified(a))

    def test_archive_roundtrip_preserves_audit_tuples(self):
        current = {'correct': False, 'proof_audit': {'reference': {'examples': [('1', '5')]}, 'reasons': ['x']}}
        archived = json.loads(json.dumps(current))
        self.assertTrue(same_archived_audit(current, archived))
        archived['proof_audit']['reference']['examples'][0][1] = '6'
        self.assertFalse(same_archived_audit(current, archived))
        archived = json.loads(json.dumps(current))
        archived['correct'] = True
        self.assertFalse(same_archived_audit(current, archived))

    def test_hidden_proof_not_promoted(self):
        _, record, _, _ = self.teachers[0]
        a = self.ns['verify_gravity_natural_language'](record['prompt'], '<think>' + record['reasoning'] + '</think>Final answer: \\boxed{' + record['final_answer'] + '}', record['final_answer'], thinking_expected=True)
        self.assertFalse(a['verified'])
        self.assertFalse(ols_certified(a))

    def test_no_training_or_hosted_entrypoint(self):
        tree = ast.parse((ROOT / 'utils/run_dpo_sft_c_gravity_diagnosis.py').read_text())
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(x.endswith(('.backward', '.step', '.save_pretrained', '.train_dpo', '.train_sft')) for x in calls))
        self.assertNotIn('NVIDIA_API_KEY', (ROOT / 'utils/run_dpo_sft_c_gravity_diagnosis.py').read_text())


if __name__ == '__main__':
    unittest.main()
