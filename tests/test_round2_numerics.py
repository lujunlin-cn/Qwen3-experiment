"""CPU mathematical oracles, independent of task quality/model performance claims."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from utils import course_worker as worker
from utils import course_protocol as protocol
from utils import course_run as supervisor
from utils.round2_evaluation import mcnemar_exact, holm, paired_comparison, wilson


class Round2Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ppo=worker.notebook_api('ppo')
        cls.grpo=worker.notebook_api('grpo')
        cls.dpo=worker.notebook_api('dpo')

    def test_gae_matches_finite_horizon_returns_and_never_zeros_observed_value(self):
        values=torch.tensor([[19.,2.,3.,4.,29.]])
        rewards=torch.tensor([[0.,0.,0.,1.,0.]])
        mask=torch.tensor([[0.,1.,1.,1.,0.]])
        dones=torch.tensor([[0.,0.,0.,1.,0.]])
        adv,returns=self.ppo['compute_gae'](rewards,values,dones,mask)
        torch.testing.assert_close(returns,torch.tensor([[0.,1.,1.,1.,0.]]))
        torch.testing.assert_close(adv,torch.tensor([[0.,-1.,-2.,-3.,0.]]))
        r=type('Rollout',(),{})();r.response_mask=mask.long();r.old_values=values.clone();r.rewards=rewards.clone()
        self.ppo['course_ppo_outcome'](r,{'completion_token_ids':[1,2,3],'max_tokens':16,
                                               'correct':False,'truncated':True})
        torch.testing.assert_close(r.old_values,values)
        self.assertEqual(float(r.rewards.sum()),0.)

    def test_ppo_policy_gradient_and_clip_match_scalar_oracle(self):
        theta=torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
        old=torch.tensor([[0.,0.]],dtype=torch.float64)
        advantage=torch.tensor([[1.2,-.7]],dtype=torch.float64);mask=torch.ones_like(old)
        loss=self.ppo['ppo_clip_loss'](old,theta[None],advantage,mask)['policy_loss']
        loss.backward()
        expected=-advantage[0]*torch.exp(theta.detach())/2
        torch.testing.assert_close(theta.grad,expected)
        clipped=torch.tensor([[.4,-.4]],requires_grad=True)
        self.ppo['ppo_clip_loss'](old.float(),clipped,advantage.float(),mask.float())['policy_loss'].backward()
        torch.testing.assert_close(clipped.grad,torch.zeros_like(clipped))

    def test_ppo_zero_outcome_can_still_train_value(self):
        values=torch.tensor([[.3,.6]],requires_grad=True)
        mask=torch.ones_like(values)
        adv,ret=self.ppo['compute_gae'](torch.zeros_like(values),values.detach(),
                                      torch.tensor([[0.,1.]]),mask)
        self.assertTrue(bool((adv!=0).any()))
        self.ppo['value_loss'](values,ret,mask).backward()
        self.assertGreater(float(values.grad.abs().sum()),0.)

    def test_dpo_multiple_betas_sequential_gradients_equal_dense(self):
        for beta in (.05,.1,.3):
            theta=torch.tensor([.4,-.3],dtype=torch.float64,requires_grad=True)
            def score(side):
                weights=theta.new_tensor([1.,3.] if side=='chosen' else [-2.,.5])
                return (theta.sin()*weights).sum().reshape(1)
            refs=[theta.new_tensor([-.6]),theta.new_tensor([.1])]
            loss=-F.logsigmoid(beta*((score('chosen')-score('rejected'))-(refs[0]-refs[1]))).mean()
            loss.backward();expected=theta.grad.clone();theta.grad=None
            with torch.no_grad(): scores=[score('chosen'),score('rejected')]
            actual=self.dpo['course_dpo_backward'](None,scores,refs,score,beta=beta)
            self.assertAlmostEqual(actual,float(loss.detach()),places=10)
            torch.testing.assert_close(theta.grad,expected)

    def test_reward_priority_gates_and_all_group_structures(self):
        reward=self.grpo['equation_reward_components']
        def sample(final,process,fmt=True,trunc=False,n=128):
            row={'task_type':'equation_numeric','completion_token_ids':[1]*n,'max_tokens':512,
                 'truncated':trunc,'finish_reason':'length' if trunc else 'stop'}
            audit={'final_correct':final,'format_valid':fmt,'exact_trace_pass':process==1,
                   'process_checks':dict.fromkeys(('example_fraction','legal_assignment','query_calculation','answer_consistency'),process)}
            return row,audit
        for ablation in ('P0','P1','P2','P3'):
            wrong=reward(*sample(False,1),ablation=ablation)
            correct=reward(*sample(True,0,False),ablation=ablation)
            self.assertGreater(correct['combined'],wrong['combined'])
            self.assertEqual(wrong['final'],0);self.assertEqual(wrong['length'],0)
            truncated=reward(*sample(True,1,trunc=True),ablation=ablation)
            self.assertEqual(truncated['combined'],0)
        for final in (False,True):
            rewards=torch.tensor([[reward(*sample(final,0),'P0')['combined']]*8])
            adv,mean,std=self.grpo['group_advantage'](rewards)
            self.assertEqual(float(adv.abs().sum()),0.)
        partial=[reward(*sample(False,x),'P2') for x in (0.,.25,.5,1.)]
        self.assertTrue(all(r['final']==0 for r in partial))
        adv,mean,std=self.grpo['group_advantage'](torch.tensor([[r['combined'] for r in partial]]))
        self.assertGreater(float(std[0]),0);self.assertLess(abs(float(adv.sum())),1e-5)
        mixed,_,_=self.grpo['group_advantage'](torch.tensor([[0.,1.,0.,1.]]))
        theta=torch.zeros((4,3),requires_grad=True)
        self.grpo['grpo_clip_loss'](theta.detach(),theta,mixed[0],torch.ones_like(theta))['loss'].backward()
        torch.testing.assert_close(theta.grad,-mixed[0,:,None].expand_as(theta)/12)

    def test_gravity_preferences_no_truncation_cross_prompt_or_unverified_chosen(self):
        def row(i,n,correct=True,truncated=False,trace=True):
            return {'sample_id':'g1','task_type':'gravity','policy_version':'sft:fixed',
                    'completion_token_ids':[i]*n,'prompt_token_ids':[1,2],'correct':correct,
                    'truncated':truncated,'finish_reason':'length' if truncated else 'stop',
                    'format_valid':True,'exact_trace_pass':trace,'predicted':'10'}
        rows=[row(1,100),row(2,200),row(3,80,False),row(4,40,False,True)]
        b=self.dpo['gravity_preference_pairs'](rows,'B')
        self.assertEqual(len(b),1);self.assertTrue(b[0]['chosen']['correct'] and b[0]['rejected']['correct'])
        self.assertFalse(self.dpo['gravity_preference_pairs']([row(1,100,trace=False),row(3,80,False)],'B'))
        with self.assertRaises(ValueError):
            bad=copy.deepcopy(rows);bad[1]['prompt_token_ids']=[3];self.dpo['gravity_preference_pairs'](bad,'B')

    def test_optimizer_roundtrip_equals_uninterrupted_two_steps(self):
        # Fails if Adam moments are lost, even when both runs change parameters.
        def update(param,optimizer):
            optimizer.zero_grad();(param.sin().square().sum()).backward();optimizer.step()
        parameter=torch.nn.Parameter(torch.tensor([.4,-.7]));model=torch.nn.ParameterList([parameter])
        opt=worker.optimizer_for(model,{}, {'learning_rate':.03});update(parameter,opt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trainer.pt';torch.save({'optimizer':opt.state_dict()},path)
            restored=torch.nn.Parameter(parameter.detach().clone());other=torch.nn.ParameterList([restored])
            second=worker.optimizer_for(other,{'trainer_state':str(path)}, {'learning_rate':.03})
            update(parameter,opt);update(restored,second)
            torch.testing.assert_close(restored,parameter,atol=0,rtol=0)
            self.assertEqual(int(second.state[restored]['step']),2)

    def test_grpo_sequence_mean_with_unequal_response_lengths(self):
        # Oracle is mean_i mean_t surrogate; global token mean is a different objective.
        parameter=torch.tensor(.15,requires_grad=True)
        advantages=torch.tensor([-1.,1.])
        losses=[]
        for i,length in enumerate((2,7)):
            new=(parameter*(i+1)).expand(1,length)
            losses.append(self.grpo['grpo_clip_loss'](torch.zeros_like(new),new,
                         advantages[i:i+1],torch.ones_like(new))['loss'])
        (sum(losses)/2).backward()
        expected=(torch.exp(parameter.detach())-2*torch.exp(2*parameter.detach()))/2
        # The positive member at ratio exp(.3)>1.2 is clipped and has zero gradient.
        expected=torch.exp(parameter.detach())/2
        torch.testing.assert_close(parameter.grad,expected)

    def test_paired_statistics_and_provenance_guards(self):
        self.assertAlmostEqual(mcnemar_exact(0,6),.03125)
        self.assertEqual(mcnemar_exact(0,0),1.)
        self.assertEqual(holm({'a':.01,'b':.04,'c':.03}),{'a':.03,'c':.06,'b':.06})
        self.assertTrue(0< wilson(4,4)[0] <1)
        rows=[{'sample_id':str(i),'task_type':'gravity','variant':'original','max_tokens':16,
            'prompt_token_ids':[2,3],'verifier_version':'official_v3','temperature':.6,'top_p':.95,
            'top_k':20,'seed':42,'correct':i<3,'truncated':False,'completion_token_ids':[1]*8} for i in range(8)]
        after=copy.deepcopy(rows)
        for r in after:r['completion_token_ids']=[1]*4
        after[3]['correct']=True
        result=paired_comparison(rows,after,draws=500)['gravity']
        self.assertEqual(result['gained_correct'],1);self.assertEqual(result['length']['mean_after_minus_before'],-4)
        with self.assertRaises(ValueError):paired_comparison(rows,after[:-1],draws=500)
        after[0]['prompt_token_ids']=[7]
        with self.assertRaises(ValueError):paired_comparison(rows,after,draws=500)

    def test_budget_incomplete_final_never_marks_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            registered=protocol.prepare(protocol.ROOT/'configs/course_8h.json',run)
            state={'jobs':{},'methods':{},'finished_at_unix':1}
            fake=SimpleNamespace(run=run,c=registered['config'],state=state)
            fake.stage=lambda name:None
            fake.evaluation=lambda *args,**kwargs:None  # exhausted budget; no partial rows invented
            fake.train_method=lambda *args,**kwargs:None
            fake.persist=lambda:protocol.write(run/'state.json',state)
            fake.persist();supervisor.Experiment.execute(fake)
            self.assertEqual(state['status'],'incomplete')
            self.assertNotIn('finished_at_unix',state)
            self.assertEqual(len(state['missing_final_evaluations']),10)

    def test_round2_worker_paths_with_real_verifier_and_tiny_parameter_updates(self):
        # Explicit numerical fixtures; never represented as real vLLM data.
        from test_course_experiment import tiny_policy
        from test_nemo_smoke import reference_proof
        from utils import run_synthetic_smoke as smoke
        from utils.low_memory_logprobs import causal_policy_logprobs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ns=smoke.load_notebook('round2_worker_fixture_only',root/'scratch')
            seeds=protocol.read_rows(smoke.ROOT/'data/processed/tasksets/roman_gravity_equation_v1/synthetic/nemo_qwen3_06b_gpu_smoke_20261109_r1/data_designer/private_seeds.jsonl')
            base,tokenizer=tiny_policy();adapter=root/'sft_fixture';base.save_pretrained(adapter)
            for method in ('dpo','ppo','grpo'):
                output=root/method;output.mkdir()
                c=protocol.read(protocol.ROOT/'configs/course_8h.json')
                settings=c[method];settings.update(gradient_accumulation=8,mode='bf16_lora')
                task='gravity' if method=='dpo' else 'equation_numeric'
                settings['tasks']=[task]
                if method=='dpo':settings.update(preference_mode='B',epochs=2)
                else:settings.update(reward_ablation='P3',loss_reduction='sequence_mean')
                seed=next(s for s in seeds if s['task_type']==task)
                proof=reference_proof(ns,seed)
                rows=[]
                for i in range(8 if method=='grpo' else 2):
                    claim=copy.deepcopy(proof)
                    if method!='dpo' and i%2:claim['answer']='wrong'
                    import json
                    response=[4,5,6]*((34 if i%2 else 67) if method=='dpo' else (i%2+1))
                    row={'sample_id':'cpu_fixture','prompt_id':'cpu_fixture','task_type':task,
                         'prompt':seed['prompt'],'expected_answer':seed['answer'],'completion':json.dumps(claim),
                         'prompt_token_ids':[2,3],'completion_token_ids':response,'group_index':i,
                         'source':'cpu_numerical_fixture','fixture':True,'source_split':'fixture',
                         'policy_version':'sft_fixture','truncated':False,'finish_reason':'stop','max_tokens':512}
                    ids=torch.tensor([[2,3]+response])
                    with torch.no_grad():row['behavior_logprobs']=causal_policy_logprobs(base,ids,None,64)[0,1:].tolist()
                    rows.append(row)
                rollout=output/'fixture_rollout.jsonl';protocol.write_rows(rollout,rows)
                job={'method':method,'round':1,'model':'cpu_fixture','seed':42,'output':str(output),
                     'adapter':str(adapter),'sft_adapter':str(adapter),'policy_version':'sft_fixture','rollout':str(rollout)}
                # Provenance gate is tested elsewhere; this bypass is confined to this numerical fixture.
                with patch.object(worker,'load_policy',side_effect=tiny_policy),patch.object(worker,'validate_training_rows'):
                    result=worker.train_dpo(job,c) if method=='dpo' else worker.train_on_policy(job,c)
                self.assertEqual(result['status'],'completed',method)
                self.assertEqual(result['optimizer_steps'],2 if method=='dpo' else 1)
                self.assertNotEqual(result['checkpoint']['weights'],worker.artifact_fingerprint(adapter))
                if method=='dpo':self.assertEqual(protocol.read(output/'preference_coverage.json')['pairs'],1)
                else:
                    audited=protocol.read_rows(output/'reward_audit.jsonl')
                    self.assertEqual(audited[1]['reward_components']['final'],0.)
                    self.assertGreater(audited[1]['reward_components']['process'],0.)


if __name__=='__main__': unittest.main()
