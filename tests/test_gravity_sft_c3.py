"""CPU gates for native Thinking, strict arithmetic and isolated epoch selection."""
import ast, copy, json, math, unittest
from unittest.mock import patch
from utils.run_gravity_sft_c3 import (ROOT,RUN,C2_RUN,BASE,CONFIG,apis,rows,read,sha,function_hashes,
    selection_key,dual_dpo_auditor,audit_rows,pair_api)


class GravityC3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.n,cls.s=apis();cls.tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        cls.old=rows(C2_RUN/'gravity_c2_teachers_74.jsonl')
        cls.records=[cls.s['make_gravity_c3_record'](cls.n['gravity_c2_solver_reference'](r['prompt']),r['source_id'],r['source_row'],r['teacher_source_group']) for r in cls.old]

    def raw(self,i=0):
        r=self.records[i];e=self.s['encode_gravity_c3_example'](self.tok,r,4096)
        return self.tok.decode(e['input_ids'][e['prompt_length']:],skip_special_tokens=False)

    def audit(self,text=None,i=0,finish='stop'):
        r=self.records[i]
        return self.n['verify_gravity_dual_channel'](r['prompt'],text if text is not None else self.raw(i),r['final_answer'],finish)

    def test_same_74_teachers_and_numeric_renderer(self):
        self.assertEqual(len(self.records),74)
        self.assertEqual(sum(r['teacher_source_group']=='historical_r1' for r in self.records),64)
        for old,r in zip(self.old,self.records):
            for key in ['prompt','completion','reasoning','final_answer','canonical_solver_trace','source_id','source_row']:
                self.assertEqual(old[key],r[key])
            self.assertEqual(r['source'],'solver_generated_rendered')

    def test_all_74_strict_dual_certificates_pass(self):
        for i in range(74):
            a=self.audit(i=i);self.assertTrue(a['verified'],(i,a['reasons']))
            self.assertTrue(a['thinking_verified']);self.assertEqual((a['rel_tol'],a['abs_tol']),(1e-4,1e-5))
            self.assertTrue(all(a['checks'].values()))

    def test_native_template_channels_and_full_loss_mask(self):
        for i,r in enumerate(self.records):
            raw=self.raw(i);thinking,final=self.n['gravity_c3_split_channels'](raw)
            self.assertEqual(thinking,r['reasoning']);self.assertEqual(final,r['assistant_content'])
            self.assertNotIn('Fit g',final);self.assertNotIn('\\boxed',thinking)
            e=self.s['encode_gravity_c3_example'](self.tok,r,4096);p=e['prompt_length']
            self.assertEqual(e['labels'][:p],[-100]*p);self.assertEqual(e['labels'][p:],e['input_ids'][p:])
            self.assertLessEqual(len(e['input_ids']),4096)

    def test_serialization_explicit_reasoning_content_and_thinking_true(self):
        seen=[];real=self.tok.apply_chat_template
        def capture(messages,**kw):seen.append((copy.deepcopy(messages),kw));return real(messages,**kw)
        with patch.object(self.tok,'apply_chat_template',side_effect=capture):self.s['encode_gravity_c3_example'](self.tok,self.records[0])
        assistant=seen[-1][0][-1]
        self.assertEqual(assistant['reasoning_content'],self.records[0]['reasoning'])
        self.assertEqual(assistant['content'],self.records[0]['assistant_content'])
        self.assertTrue(all(kw['enable_thinking'] for _,kw in seen))

    def test_bare_boxed_final_accepted(self):
        self.assertTrue(self.audit(self.raw().replace('Final answer: ',''))['verified'])

    def test_wrong_answer_claim_inside_thinking_rejected(self):
        self.assertFalse(self.audit(self.raw().replace('</think>','Final answer: 9999\n</think>'))['verified'])

    def test_legacy_v1_2_results_are_preserved(self):
        for i in range(74):
            a=self.audit(i=i)
            self.assertFalse(a['legacy_v1_2_final_only']['verified'])
            self.assertEqual(a['legacy_v1_2_final_only']['version'],'gravity-natural-language-v1.2')

    def test_missing_close_rejected(self):self.assertFalse(self.audit(self.raw().replace('</think>',''))['verified'])
    def test_multiple_close_rejected(self):self.assertFalse(self.audit(self.raw()+'</think>')['verified'])
    def test_empty_thinking_rejected(self):self.assertFalse(self.audit('</think>'+self.records[0]['assistant_content'])['verified'])
    def test_length_rejected(self):self.assertFalse(self.audit(finish='length')['verified'])
    def test_reopened_thinking_rejected(self):self.assertFalse(self.audit(self.raw()+'<think>')['verified'])
    def test_duplicate_final_answer_rejected(self):self.assertFalse(self.audit(self.raw()+' \\boxed{1}')['verified'])
    def test_conflicting_unboxed_final_rejected(self):self.assertFalse(self.audit(self.raw().replace('<|im_end|>','')+' answer=9999')['verified'])
    def test_correct_final_does_not_rescue_wrong_g(self):
        raw=self.raw().replace('The fitted g is ','The fitted g is 9999; previous ')
        self.assertFalse(self.audit(raw)['verified'])
    def test_correct_final_does_not_rescue_wrong_raw(self):
        raw=self.raw().replace('The unrounded Query distance is ','The unrounded Query distance is 9999; previous ')
        self.assertFalse(self.audit(raw)['verified'])
    def test_wrong_query_time_rejected(self):
        raw=self.raw().replace('to Query t=','to Query t=9999; previous ')
        self.assertFalse(self.audit(raw)['verified'])
    def test_wrong_final_does_not_pass_correct_thinking(self):
        r=self.records[0];raw=self.raw().replace(r['assistant_content'],'Final answer: \\boxed{9999}')
        self.assertFalse(self.audit(raw)['verified'])

    def test_wrong_contribution_rejected(self):
        raw=self.raw();line=next(l for l in raw.splitlines() if '^2*' in l and l[0].isdigit())
        raw=raw.replace(line,line.split('=')[0]+'=9999')
        self.assertFalse(self.audit(raw)['verified'])
    def test_omitted_contribution_rejected(self):
        raw=self.raw();line=next(l for l in raw.splitlines() if '^4 =' in l)
        self.assertFalse(self.audit(raw.replace(line,'Calculation omitted.'))['verified'])
    def test_wrong_sum_rejected(self):
        raw=self.raw();line=next(l for l in raw.splitlines() if l.startswith('Weighted data sum:'))
        raw=raw.replace(line,line.rsplit(' = ',1)[0]+' = 9999')
        self.assertFalse(self.audit(raw)['verified'])
    def test_wrong_ols_numeric_substitution_rejected(self):
        raw=self.raw();lines=raw.splitlines();i=lines.index('Numerical OLS substitution:')+1
        lines[i]=lines[i].replace('2*','1*',1)
        self.assertFalse(self.audit('\n'.join(lines))['verified'])
    def test_keywords_without_numeric_certificate_rejected(self):
        r=self.records[0];trace=r['canonical_solver_trace']
        text=f"Fit g = 2*sum(t^2*d)/sum(t^4) using all public examples. The fitted g is {trace['g_ols']}; apply distance = 0.5*g*t^2 to Query t={trace['query_t']}.\nThe unrounded Query distance is {trace['query_raw']}.\n</think>"+r['assistant_content']
        a=self.audit(text);self.assertFalse(a['verified']);self.assertFalse(a['checks']['all_observation_terms'])
    def test_single_point_substitution_rejected(self):
        raw=self.raw();lines=raw.splitlines();i=lines.index('Numerical OLS substitution:')+1
        lines[i]='2*45.86/(3.03)^2 = 9.990094729344825'
        self.assertFalse(self.audit('\n'.join(lines))['verified'])
    def test_paraphrase_is_not_exact_string_comparison(self):
        raw=self.raw().replace('Read all public time-distance examples:','Read all observations:').replace('Weighted data sum:','Weighted data sum:  ')
        self.assertTrue(self.audit(raw)['verified'])
    def test_unsupported_or_malicious_arithmetic_rejected(self):
        with self.assertRaises((ValueError,SyntaxError)):
            self.n['_gravity_nl_eval']('__import__("os").system("false")')
        self.assertFalse(self.audit(self.raw().replace('Numerical OLS substitution:','Numerical OLS substitution:\n1/0=2'))['verified'])
    def test_no_teacher_truncation_fallback(self):
        with self.assertRaises(ValueError):self.s['encode_gravity_c3_example'](self.tok,self.records[0],32)
    def test_no_replay_allowed(self):
        with self.assertRaises(ValueError):self.s['train_gravity_c3_branch'](None,None,[],read(CONFIG)['sft'])

    def test_same_optimizer_settings_as_c2(self):
        old=read(ROOT/'configs/gravity_sft_c2_20261116.json')['sft'];new=read(CONFIG)['sft']
        for key in ['mode','epochs','learning_rate','weight_decay','adam_epsilon','micro_batch_size','gradient_accumulation','max_grad_norm','max_length','projection_chunk_size','seed','enable_thinking']:
            self.assertEqual(old[key],new[key])
        self.assertFalse(read(CONFIG)['replay'])

    def test_solver_legacy_verifier_renderer_and_default_sft_unchanged(self):
        for file,names in [('01_Synthetic_Data_Engineering.ipynb',{'solve_gravity_prompt','_gravity_verifier_fit','gravity_c2_solver_reference','verify_gravity_natural_language'}),
                           ('03_SFT.ipynb',{'render_gravity_c2_final_proof','encode_chat_example','course_sft_loss_sum','choose_sft_records','build_full_trace_records'})]:
            self.assertEqual(function_hashes(read(RUN/'source_before_repair'/file),names),function_hashes(read(ROOT/file),names))

    def test_training_objective_only_gained_optional_epoch_hook(self):
        def extract(book):
            for cell in book['cells']:
                if cell['cell_type']!='code':continue
                for n in ast.parse(''.join(cell['source'])).body:
                    if isinstance(n,ast.FunctionDef) and n.name=='train_augmentation_branch':return n
        old=extract(read(RUN/'source_before_repair/03_SFT.ipynb'));new=extract(read(ROOT/'03_SFT.ipynb'))
        new.args.args.pop();new.args.defaults.pop()
        epoch_loop=next(n for n in new.body if isinstance(n,ast.For));epoch_loop.body.pop()
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_protected_c2_history_and_other_algorithms_unchanged(self):
        protected=read(RUN/'initial_state.json')['protected_hashes']
        for p,h in protected.items():self.assertEqual(sha(ROOT/p),h,p)

    def test_selection_is_frozen_and_uses_no_final_or_regression(self):
        def m(p,a,t=0,length=10):return {'proof_pass':p,'correct':a,'truncated':t,'mean_tokens':length}
        self.assertGreater(selection_key(1,m(1,1)),selection_key(2,m(0,48)))
        self.assertGreater(selection_key(1,m(0,3)),selection_key(2,m(0,2)))
        self.assertGreater(selection_key(1,m(0,3)),selection_key(2,m(0,3)))
        self.assertEqual(read(CONFIG)['selection']['locked_final_uses'],1)
        self.assertFalse(read(CONFIG)['regression']['selection_use'])

    def test_dual_auditor_is_scoped_and_original_restored(self):
        import utils.dpo_gravity_worker as worker
        old=worker.audit_nl_rows
        with dual_dpo_auditor():self.assertIs(worker.audit_nl_rows,audit_rows)
        self.assertIs(worker.audit_nl_rows,old)
        with self.assertRaises(RuntimeError):
            with dual_dpo_auditor():raise RuntimeError('probe')
        self.assertIs(worker.audit_nl_rows,old)

    def test_same_dpo_loss_settings_and_pair_thresholds(self):
        self.assertEqual(read(CONFIG)['dpo'],read(ROOT/'configs/gravity_sft_c2_20261116.json')['dpo'])
        r=self.records[0];base={'task_type':'gravity','sample_id':'x','prompt_token_ids':[1,2],
            'policy_version':'C3:hash','correct':True,'finish_reason':'stop','truncated':False,'proof_audit':{'verified':True}}
        a={**base,'group_index':0,'completion_token_ids':[1]*200}
        b={**base,'group_index':1,'completion_token_ids':[1]*300}
        self.assertEqual(len(pair_api()([a,b])),1)
        self.assertEqual(pair_api()([a,{**b,'proof_audit':{'verified':False}}]),[])
        self.assertEqual(pair_api()([a,{**b,'correct':False}]),[])
        self.assertEqual(pair_api()([a,{**b,'completion_token_ids':[1]*240}]),[])


if __name__=='__main__':unittest.main()
