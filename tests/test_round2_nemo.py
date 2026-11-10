"""Real NeMo SDK + mock HTTP, not real model generation or quality gains."""
import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from utils import run_synthetic_smoke as smoke
from test_nemo_smoke import reference_proof


class Round2NeMoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();root=Path(cls.tmp.name)
        cls.seed_run=smoke.prepare(run_id='round2_cpu_fixture_only',artifact_dir=root/'seed')
        cls.ns=smoke.load_notebook('round2_cpu_fixture_only',cls.seed_run)
        cls.seeds=smoke.rows(cls.seed_run/'data_designer/private_seeds.jsonl')

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_independent_process_partial_signal_without_false_final_reward(self):
        seed=next(r for r in self.seeds if r['task_type']=='equation_numeric')
        proof=reference_proof(self.ns,seed)
        clean=self.ns['audit_equation_process'](seed,json.dumps(proof))
        self.assertTrue(clean['exact_trace_pass']);self.assertEqual(clean['process_checks']['example_fraction'],1)
        wrong=copy.deepcopy(proof);wrong['example_outputs'][0]='999999';wrong['answer']='wrong'
        audit=self.ns['audit_equation_process'](seed,json.dumps(wrong))
        self.assertFalse(audit['final_correct']);self.assertFalse(audit['exact_trace_pass'])
        self.assertGreater(audit['process_checks']['example_fraction'],0)
        self.assertFalse(audit['process_checks']['query_calculation'])
        self.assertFalse(audit['process_checks']['answer_consistency'])
        bad=json.dumps(proof)[:-1]+',"answer":"wrong"}'
        self.assertFalse(self.ns['audit_equation_process'](seed,bad)['format_valid'])
        original=self.ns['solve_equation_numeric_prompt']
        def forbidden(*a,**kw):raise AssertionError('Verifier called Solver')
        self.ns['solve_equation_numeric_prompt']=forbidden
        try:self.assertTrue(self.ns['audit_equation_process'](seed,json.dumps(proof))['exact_trace_pass'])
        finally:self.ns['solve_equation_numeric_prompt']=original

    def test_all_arms_actual_sdk_request_fields_schema_and_raw_screening(self):
        from data_designer.interface import DataDesigner
        seeds={r['seed_id']:r for r in self.seeds};ns=self.ns;requests=[];errors=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                content=payload['messages'][-1]['content']
                text=''.join(x.get('text','') for x in content) if isinstance(content,list) else content
                seed=seeds[text.split('SEED_ID=',1)[1].splitlines()[0]]
                requests.append(payload)
                if seed['prompt'] not in text or '{{' in text or '{%' in text:errors.append('rendering failed')
                value=ns['SMOKE_PREFIXES'][0]+'\n'+seed['prompt'] if 'safe wording variant' in text else json.dumps(reference_proof(ns,seed))
                body={'id':'fixture','object':'chat.completion','created':1,'model':'Qwen/Qwen3-0.6B',
                    'choices':[{'index':0,'message':{'role':'assistant','content':value},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}
                data=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                for arm in ('R0','A','B','C'):
                    requests.clear();records=[]
                    public=[{k:s[k] for k in ('seed_id','task_type','prompt')} for s in self.seeds]
                    if arm=='C':
                        for row,seed in zip(public,self.seeds):row['solver_context']=json.dumps(ns['smoke_solver_context'](seed))
                    groups=[(t,[r for r in public if r['task_type']==t]) for t in ns['SMOKE_TASKS']] if arm=='B' else [(None,public)]
                    capture=root/f'{arm}_requests.jsonl'
                    with smoke.capture_local_requests(f'http://127.0.0.1:{server.server_port}/v1',capture) as proxy:
                        for task,rows in groups:
                            shard=root/(arm+'_'+str(task));shard.mkdir()
                            seed_path=shard/'seeds.jsonl';ns['_write_jsonl'](seed_path,rows)
                            builder,provider=ns['build_round2_data_designer'](seed_path,proxy,arm,task)
                            designer=DataDesigner(artifact_path=shard/'sdk',model_providers=[provider]);ns['configure_smoke_runtime'](designer)
                            designer.validate(builder)
                            result=designer.preview(config_builder=builder,num_records=len(rows))
                            records.extend(result.dataset.to_dict(orient='records'))
                    transport=smoke.attach_transport(records,capture)
                    self.assertEqual(transport['http_requests'],18,arm);self.assertEqual(errors,[],arm)
                    summary=ns['screen_data_designer_records'](records,self.seeds,root/(arm+'_screen'),require_transport=True)
                    self.assertEqual(summary['accepted'],18,(arm,summary))
                    if arm=='B':self.assertEqual(sum('response_format' in r for r in requests),9)
                    if arm=='C':self.assertTrue(any('SOLVER_CONTEXT_FROM_PUBLIC_PROMPT=' in str(r['messages']) for r in requests))
        finally:server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':unittest.main()
