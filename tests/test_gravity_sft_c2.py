"""Real-data CPU gates for final-channel Gravity C2 and conditional DPO."""
import ast, copy, json, math, unittest
from unittest.mock import patch
from utils.run_gravity_sft_c2 import ROOT,RUN,BASE,CONFIG,apis,actual_teachers,nl_api,function_hashes,pair_api,read,rows


class GravityC2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.n,cls.s=apis();cls.tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        cls.nl=nl_api();cls.teachers=actual_teachers()[0]

    def record(self,index=0):
        group,record,source,_=self.teachers[index]
        sample=self.n['gravity_c2_solver_reference'](record['prompt'])
        return self.s['make_gravity_c2_record'](sample,record['source_id'],source['source_row'],group)

    def test_exactly_74_actual_train_inputs(self):
        self.assertEqual(len(self.teachers),74)
        self.assertEqual(sum(x[0]=='historical_r1' for x in self.teachers),64)
        self.assertEqual(sum(x[0]=='added_real_c' for x in self.teachers),10)

    def test_every_rich_target_passes_unchanged_verifier(self):
        for index in range(74):
            r=self.record(index)
            a=self.nl['verify_gravity_natural_language'](r['prompt'],r['completion'],r['final_answer'])
            self.assertTrue(a['verified'],(r['source_id'],a['reasons']))
            self.assertEqual(r['source'],'solver_generated_rendered')

    def test_each_observation_and_numerical_ols_terms_shown(self):
        for index in range(74):
            r=self.record(index);trace=r['canonical_solver_trace']
            for i,term in enumerate(trace['public_observations'],1):
                self.assertIn(f"Observation {i}: t = {term['time']}s, distance = {term['distance']} m",r['completion'])
                self.assertIn(term['time']+'^2*'+term['distance'],r['completion'])
                self.assertIn(term['time']+'^4',r['completion'])
            self.assertIn('Numerical OLS substitution:',r['completion'])
            self.assertIn('sum(t^2*d)/sum(t^4)',r['completion'])

    def test_final_channel_contains_entire_proof(self):
        for index in range(74):
            r=self.record(index);encoded=self.s['encode_gravity_c2_final_example'](self.tok,r,4096)
            response=self.tok.decode(encoded['input_ids'][encoded['prompt_length']:],skip_special_tokens=False)
            hidden,final=response.split('</think>',1)
            self.assertNotIn('Fit g',hidden);self.assertIn(r['completion'],final)
            self.assertTrue(self.nl['verify_gravity_natural_language'](r['prompt'],response,r['final_answer'],thinking_expected=True)['verified'])

    def test_all_mathematical_tokens_are_supervised(self):
        for index in range(74):
            r=self.record(index);e=self.s['encode_gravity_c2_final_example'](self.tok,r,4096);p=e['prompt_length']
            self.assertEqual(e['labels'][:p],[-100]*p);self.assertEqual(e['labels'][p:],e['input_ids'][p:])
            self.assertLessEqual(len(e['input_ids']),4096)

    def test_serializer_never_sends_reasoning_content(self):
        r=self.record();real=self.tok.apply_chat_template;seen=[]
        def inspect(messages,**kw):
            seen.extend(copy.deepcopy(messages));return real(messages,**kw)
        with patch.object(self.tok,'apply_chat_template',side_effect=inspect):
            self.s['encode_gravity_c2_final_example'](self.tok,r,4096)
        assistants=[r for r in seen if r['role']=='assistant']
        self.assertEqual(len(assistants),1);self.assertNotIn('reasoning_content',assistants[0])
        self.assertEqual(assistants[0]['content'],r['completion'])

    def test_no_truncation_fallback(self):
        with self.assertRaises(ValueError):self.s['encode_gravity_c2_final_example'](self.tok,self.record(),16)

    def test_final_box_matches_original_teacher(self):
        for i,(_,old,_,_) in enumerate(self.teachers):
            r=self.record(i);self.assertEqual(r['final_answer'],old['final_answer'])
            self.assertTrue(r['completion'].endswith('Final answer: \\boxed{'+old['final_answer']+'}'))

    def test_renderer_is_deterministic_and_uses_no_hidden_rule(self):
        r=self.record();self.assertEqual(r,self.record())
        self.assertNotIn('hidden_rule',r['completion']);self.assertNotIn('ground truth',r['completion'])

    def test_wrong_numeric_ols_sum_rejected(self):
        r=self.record();text=r['completion'];line=next(x for x in text.splitlines() if x.startswith('Weighted data sum:'))
        text=text.replace(line,line.rsplit(' = ',1)[0]+' = 999999')
        self.assertFalse(self.nl['verify_gravity_natural_language'](r['prompt'],text,r['final_answer'])['verified'])

    def test_wrong_fitted_g_rejected(self):
        r=self.record();text=r['completion'].replace('The fitted g is ','The fitted g is 999999; prior value ')
        self.assertFalse(self.nl['verify_gravity_natural_language'](r['prompt'],text,r['final_answer'])['verified'])

    def test_thinking_cannot_rescue_final_only_answer(self):
        r=self.record();text='<think>'+r['completion']+'</think>Final answer: \\boxed{'+r['final_answer']+'}'
        self.assertFalse(self.nl['verify_gravity_natural_language'](r['prompt'],text,r['final_answer'],thinking_expected=True)['verified'])

    def test_legacy_default_encoder_and_math_loss_unchanged(self):
        old=read(RUN/'source_before_repair/03_SFT.ipynb');now=read(ROOT/'03_SFT.ipynb')
        names={'encode_chat_example','course_sft_loss_sum','build_full_trace_records','choose_sft_records'}
        self.assertEqual(function_hashes(old,names),function_hashes(now,names))

    def test_independent_nl_verifier_functions_unchanged(self):
        names={'gravity_nl_public_reference','gravity_nl_final_section','_gravity_nl_normalize','_gravity_nl_eval','verify_gravity_natural_language'}
        self.assertEqual(function_hashes(read(RUN/'source_before_repair/01_Synthetic_Data_Engineering.ipynb'),names),function_hashes(read(ROOT/'01_Synthetic_Data_Engineering.ipynb'),names))

    def test_training_settings_match_existing_sft_c(self):
        old=read(ROOT/'configs/equation_sft_augmentation_20261116.json');new=read(CONFIG)['sft']
        keys=['mode','epochs','learning_rate','weight_decay','adam_epsilon','micro_batch_size','gradient_accumulation','max_grad_norm','max_length','projection_chunk_size','seed']
        self.assertEqual({k:old[k] for k in keys},{k:new[k] for k in keys})

    def test_dpo_loss_and_ppo_grpo_files_unchanged(self):
        from utils.dpo_gravity_worker import sha
        protected=read(RUN/'initial_state.json')['protected_hashes']
        for p in ['04_DPO.ipynb','05_PPO.ipynb','06_GRPO.ipynb','utils/course_worker.py','utils/low_memory_logprobs.py']:
            self.assertEqual(sha(ROOT/p),protected[p])

    def test_dpo_config_gate_and_original_preferences(self):
        c=read(CONFIG);self.assertEqual(c['dpo'],read(ROOT/'configs/dpo_gravity_sft_c_20261116.json')['dpo'])
        self.assertEqual(c['auto_train_dpo_only_when_pairs_at_least'],16)
        self.assertEqual(pair_api()([]),[])

    def test_only_74_gravity_data_allowed_by_sft_wrapper(self):
        with self.assertRaises(ValueError):self.s['train_gravity_c2_branch'](None,None,[],read(CONFIG)['sft'])


if __name__=='__main__':unittest.main()
