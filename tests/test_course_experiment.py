"""CPU-only protocol and real tiny-Qwen/PEFT replay; never GPU evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from utils import course_protocol as protocol
from utils import course_worker as worker
from utils import course_run as supervisor
from utils.low_memory_logprobs import causal_policy_logprobs

CONFIG=protocol.ROOT/'configs/course_8h.json'

class Tokenizer:
    pad_token_id=0
    eos_token_id=1
    def apply_chat_template(self,messages,**kwargs):
        prefix='<user>'+messages[0]['content']+'</user><think>'
        if len(messages)==1: return prefix
        return prefix+messages[1]['reasoning_content']+'</think>'+messages[1]['content']+'<eos>'
    def __call__(self,text,**kwargs): return SimpleNamespace(input_ids=[2+ord(c)%62 for c in text])
    def save_pretrained(self,path): protocol.write(Path(path)/'tokenizer_fixture.json',{'fixture':True})

def tiny_policy(model_path=None,mode=None,adapter=None):
    torch.manual_seed(71)
    cfg=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                   num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=4096,
                   tie_word_embeddings=True,attention_dropout=0.0)
    cfg._attn_implementation='sdpa'
    base=Qwen3ForCausalLM(cfg)
    base.config.use_cache=False
    if adapter:
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=True)
    else:
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();worker.training_mode(model)
    return model,Tokenizer()

class ProtocolTests(unittest.TestCase):
    def test_panels_metrics_and_resume_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            p=protocol.prepare(CONFIG,run)
            self.assertFalse(p['holdout_used'])
            self.assertEqual(protocol.prepare(CONFIG,run),p)
            tune=protocol.read_rows(run/'panels/tune.jsonl');final=protocol.read_rows(run/'panels/final.jsonl')
            self.assertFalse({r['id'] for r in tune}&{r['id'] for r in final})
            for method in ('ppo','grpo'):
                self.assertEqual({r['task_type'] for r in protocol.read_rows(run/f'panels/{method}_1.jsonl')},{'equation_numeric'})
            rows=[{'task_type':'gravity','correct':True,'truncated':False,'completion_token_ids':[1,2]},
                  {'task_type':'gravity','correct':False,'truncated':True,'completion_token_ids':[1]*6}]
            m=protocol.metrics(rows)['gravity']
            self.assertEqual((m['accuracy'],m['truncation_rate'],m['mean_output_tokens']),(.5,.5,4))
            (run/'panels/tune.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen'): protocol.check_protocol(run)

    def test_priors_do_not_require_target_and_keep_format(self):
        prior=protocol.priors()
        self.assertIn('1=I',prior['numeral_system']);self.assertIn('10=X',prior['numeral_system'])
        self.assertIn('g = 2*d/t^2',prior['gravity'])
        self.assertIn('flip_absdiff_suffix',prior['equation_numeric'])
        self.assertEqual(protocol.prompt_text('QUESTION','gravity','original',prior),'QUESTION')
        rows=[{'task_type':'equation_numeric','completion':r'Final answer: \boxed{17/}', 'expected_answer':'17/','truncated':False}]
        self.assertTrue(worker.verify_rows(rows)[0]['correct'])

    def test_stopping_is_validation_based(self):
        stop={'minimum_rounds':2,'patience':2,'min_score_gain':.01}
        self.assertEqual(protocol.advance_stop([],0.1,stop),(True,False))
        self.assertEqual(protocol.advance_stop([.1,.2,.2],.2,stop),(False,True))
        self.assertEqual(protocol.advance_stop([.1,.2],.25,stop),(True,False))
        self.assertEqual(protocol.advance_stop([.1,.105],.111,stop),(True,False))

    def test_guard_limit_and_deadline_use_only_child_group(self):
        c=protocol.read(CONFIG)
        with tempfile.TemporaryDirectory() as d:
            readings=iter([{'used_bytes':8_000_000_001}])
            with patch.object(supervisor,'idle_device',return_value={'used_bytes':0,'uuid':'fixture','name':'fixture'}), \
                 patch.object(supervisor,'gpu_memory',side_effect=lambda gpu:next(readings)):
                r=supervisor.guard_job(['/bin/sleep','5'],Path(d),'0',c,__import__('time').time()+10)
            self.assertEqual(r['status'],'memory_limit_exceeded')
            self.assertEqual(r['memory_check'],'failed')
        with tempfile.TemporaryDirectory() as d:
            with patch.object(supervisor,'idle_device',return_value={'used_bytes':0,'uuid':'fixture','name':'fixture'}), \
                 patch.object(supervisor,'gpu_memory',return_value={'used_bytes':100}):
                r=supervisor.guard_job(['/bin/sleep','5'],Path(d),'0',c,__import__('time').time()-.1)
            self.assertEqual(r['status'],'budget_exhausted')

    def test_full_supervisor_protocol_and_idempotent_resume_without_gpu(self):
        import sys
        import time
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';model=root/'model';model.mkdir()
            (model/'model.safetensors').write_bytes(b'cpu supervisor fixture only')
            protocol.write(model/'config.json',{'fixture':True})
            protocol.prepare(CONFIG,run)
            args=SimpleNamespace(run_dir=run,config=CONFIG,model=model,
                                 inference_python=Path(sys.executable),training_python=Path(sys.executable),gpu='0')
            def fake_guard(command,output,gpu,c,deadline):
                output=Path(output);job=protocol.read(output/'job.json')
                if job['kind']=='training':
                    adapter=output/'adapter';adapter.mkdir()
                    (adapter/'adapter_model.safetensors').write_bytes(str(output).encode())
                    protocol.write(adapter/'adapter_config.json',{'fixture':True})
                    (output/'trainer.pt').write_bytes(b'fixture')
                    checkpoint={'adapter_path':str(adapter),'policy_version':output.name,
                        'trainer_state':str(output/'trainer.pt'),'value_head':None}
                    result={'status':'completed','checkpoint':checkpoint,'optimizer_steps':1}
                else:
                    panel=protocol.read_rows(job['panel'])
                    rows=[{'sample_id':r['id'],'task_type':r['task_type'],'correct':True,'truncated':False,
                           'completion_token_ids':[1,2,3],'variant':job['variant'],
                           'max_tokens':job.get('cap') or c['eval_caps'][r['task_type']]} for r in panel]
                    protocol.write_rows(output/'rows.jsonl',rows)
                    result={'status':'completed','metrics':protocol.metrics(rows),'output_tokens_per_second':10}
                protocol.write(output/'result.json',result)
                return {'status':'completed','wall_seconds':1,'device_peak_sampled_bytes':1024,
                        'memory_check':'sampled_pass','returncode':0}
            with patch.object(supervisor,'guard_job',side_effect=fake_guard) as guard:
                supervisor.Experiment(args).execute()
                calls=guard.call_count
                supervisor.Experiment(args).execute()
                self.assertEqual(guard.call_count,calls)
            result=protocol.read(run/'report.json')
            self.assertEqual(result['status'],'complete')
            self.assertFalse(result['continuous_memory_bound_verified'])
            self.assertEqual(result['missing_final_evaluations'],[])
            self.assertEqual({m['stop_reason'] for m in result['methods'].values()},{'validation_plateau'})
            self.assertTrue(any(r['method']=='prompt_prior' for r in result['comparisons']))
            self.assertIn('grpo:rollout',result['phase_wall_seconds'])

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns={m:worker.notebook_api(m) for m in ('sft','dpo','ppo','grpo')}

    def test_decoder_checkpointing_recomputes_without_changing_gradients(self):
        model, _ = tiny_policy()
        model.eval()
        ids = torch.tensor([[2, 3, 4, 5, 6]])
        calls = []
        layer = model.get_base_model().model.layers[0].self_attn.q_proj
        hook = layer.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            causal_policy_logprobs(model, ids, None, 8).sum().backward()
            baseline_calls = len(calls)
            grads = {n:p.grad.clone() for n,p in model.named_parameters()
                     if p.requires_grad and p.grad is not None}
            model.zero_grad(set_to_none=True)
            calls.clear()
            worker.training_mode(model)
            causal_policy_logprobs(model, ids, None, 8).sum().backward()
            self.assertGreater(len(calls), baseline_calls)
            for name, parameter in model.named_parameters():
                if name in grads:
                    torch.testing.assert_close(parameter.grad, grads[name], rtol=1e-5, atol=1e-6)
        finally:
            hook.remove()

    def test_dpo_sequential_backward_is_exact(self):
        theta=torch.nn.Parameter(torch.tensor([.3,-.4],dtype=torch.float64))
        def score(side):
            return (theta.sin()*torch.tensor([1.,2.] if side=='chosen' else [-3.,.5])).sum().reshape(1)
        reference=[torch.tensor([-.2]),torch.tensor([.7])]
        chosen,rejected=score('chosen'),score('rejected')
        loss=-F.logsigmoid(.1*((chosen-rejected)-(reference[0]-reference[1]))).mean()
        loss.backward();expected=theta.grad.clone();theta.grad=None
        with torch.no_grad(): values=[score('chosen'),score('rejected')]
        actual=self.ns['dpo']['course_dpo_backward'](None,values,reference,score)
        self.assertAlmostEqual(actual,float(loss.detach()),places=9)
        torch.testing.assert_close(theta.grad,expected,rtol=1e-7,atol=1e-9)

    def test_preference_and_reward_dominance(self):
        def row(k,n,good=True,trunc=False):
            return {'sample_id':'x','task_type':'gravity','completion_token_ids':[k]*n,'completion':str(k),
                    'prompt_token_ids':[1,2],'correct':good,'truncated':trunc,'max_tokens':512}
        rows=[row(3,100),row(4,300),row(5,512,False,True)]
        pairs=self.ns['dpo']['course_preference_pairs'](rows)
        self.assertEqual({p['preference_reason'] for p in pairs},{'correctness','correct_short_over_correct_long'})
        self.assertTrue(any(p['rejected']['truncated'] for p in pairs))
        reward=self.ns['grpo']['course_efficiency_reward']
        self.assertGreater(reward(rows[0]),reward(rows[1]));self.assertEqual(reward(rows[2]),0)
        advantages,_,std=self.ns['grpo']['group_advantage'](torch.tensor([[reward(r) for r in rows[:2]]]))
        self.assertGreater(float(std[0]),0);self.assertGreater(float(advantages[0,0]),0)

    def test_sft_projection_matches_dense_loss_and_gradients(self):
        model,_=tiny_policy()
        ids=torch.tensor([[2,3,4,5,6]])
        labels=ids.clone();labels[:,:2]=-100
        expected=F.cross_entropy(model(ids).logits[:,:-1].float().reshape(-1,64),labels[:,1:].reshape(-1),reduction='sum')
        expected.backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        model.zero_grad(set_to_none=True)
        actual,n=self.ns['sft']['course_sft_loss_sum'](model,{'input_ids':ids,'labels':labels,'attention_mask':torch.ones_like(ids)},2)
        actual.backward();self.assertEqual(n,3)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        for name,p in model.named_parameters():
            if name in grads: torch.testing.assert_close(p.grad,grads[name],rtol=1e-5,atol=1e-6)

    def test_two_round_real_tiny_model_replay_all_methods(self):
        c=protocol.read(CONFIG)
        for method in ('sft','dpo','ppo','grpo'):
            c[method]['gradient_accumulation']=2
        with tempfile.TemporaryDirectory() as directory,patch.object(worker,'load_policy',side_effect=tiny_policy):
            run=Path(directory);(run/'panels').mkdir()
            traces=[]
            for task,trace in [('numeral_system',{'query_number':4,'components':[{'arabic':4,'roman':'IV'}]}),
                               ('gravity',{'g_ols':10,'query_t':2,'query_raw':20}),
                               ('equation_numeric',{'candidate_rules_by_operator':{'+':['add']},'query_outputs':['4']})]:
                traces.append({'task_type':task,'prompt':'Toy only','answer':'4','trace':trace,
                    'source_id':task,'verification_status':'accepted'})
            protocol.write_rows(run/'panels/sft.jsonl',traces)
            def job(method,k,parent=None,sft=None):
                out=run/f'{method}{k}';out.mkdir()
                return {'method':method,'round':k,'model':'cpu_fixture','seed':42+k,'run':str(run),'output':str(out),
                    'adapter':parent['adapter_path'] if parent else None,
                    'trainer_state':parent.get('trainer_state') if parent else None,
                    'value_head':parent.get('value_head') if parent else None,
                    'policy_version':parent['policy_version'] if parent else 'base',
                    'sft_adapter':sft['adapter_path'] if sft else None}
            sft1=worker.sft_epoch(job('sft',1),c)['checkpoint']
            sft=worker.sft_epoch(job('sft',2,sft1),c)['checkpoint']
            for method in ('dpo','ppo','grpo'):
                parent=None
                for k in (1,2):
                    j=job(method,k,parent or sft,sft)
                    if k==1: j['trainer_state']=None;j['value_head']=None
                    model,_=tiny_policy(adapter=j['adapter'])
                    rows=[]
                    for i in range(8):
                        response=[4,5,6,7] if i%2 else [8,9,10]
                        r={'sample_id':'fixture','prompt_id':'fixture','task_type':'equation_numeric','prompt':'Toy only',
                           'expected_answer':'4','completion':str(i),'prompt_token_ids':[2,3],
                           'completion_token_ids':response,'correct':bool(i%2),'truncated':not bool(i%2),
                           'finish_reason':'stop' if i%2 else 'length','source_split':'train',
                           'source':'vllm_rollout','fixture':False,'max_tokens':16,'group_index':i,
                           'policy_version':j['policy_version']}
                        ids=torch.tensor([r['prompt_token_ids']+response])
                        with torch.no_grad(): r['behavior_logprobs']=causal_policy_logprobs(model,ids,None,2)[0,1:].tolist()
                        rows.append(r)
                    rollout=run/f'{method}{k}.jsonl';protocol.write_rows(rollout,rows);j['rollout']=str(rollout)
                    result=worker.train_dpo(j,c) if method=='dpo' else worker.train_on_policy(j,c)
                    self.assertEqual(result['status'],'completed',method)
                    self.assertGreater(result['optimizer_steps'],0)
                    parent=result['checkpoint']
                    self.assertNotEqual(parent['weights'],worker.artifact_fingerprint(j['adapter']))
                    self.assertEqual(parent['sft_reference'],sft['adapter_path'])
                reloaded,_=tiny_policy(adapter=parent['adapter_path'])
                self.assertTrue(any(p.requires_grad for p in reloaded.parameters()))

if __name__=='__main__': unittest.main()
