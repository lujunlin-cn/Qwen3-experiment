"""Actual NeMo SDK over a mock loopback OpenAI server; no model/GPU evidence."""
import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

from utils import run_synthetic_smoke as smoke


def reference_proof(ns,seed):
    task=seed['task_type'];trace=seed['trace']
    if task=='numeral_system':
        return {'task_type':task,'query':trace['query_number'],'components':copy.deepcopy(trace['components']),'answer':seed['answer']}
    if task=='gravity':
        return {'task_type':task,'g_estimates':list(trace['g_estimates']),'g_ols':trace['g_ols'],
                'query_t':trace['query_t'],'query_distance':trace['query_raw'],'answer':seed['answer']}
    examples,query=ns['_parse_numeric_prompt'](seed['prompt'])
    rules=trace['selected_assignment']['rules']
    return {'task_type':task,'rules':rules,'example_outputs':[ns['_numeric_verify_output'](rules[e[2]],e) for e,_ in examples],
            'query_output':ns['_numeric_verify_output'](rules[query[2]],query),'answer':seed['answer']}


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.prepared_run=smoke.prepare(run_id='nemo_unittest_fixture_only',artifact_dir=Path(cls.directory.name)/'prepared')
        cls.ns=smoke.load_notebook('nemo_unittest_fixture_only',cls.prepared_run)
        cls.seeds=smoke.rows(cls.prepared_run/'data_designer/private_seeds.jsonl')

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def test_seed_manifest_and_empty_accepted_status(self):
        smoke.verify_prepared_run(self.prepared_run)
        self.assertEqual(len(self.seeds),9)
        public=smoke.rows(self.prepared_run/'data_designer/public_seeds.jsonl')
        self.assertTrue(all(set(r)=={'seed_id','task_type','prompt'} for r in public))
        with self.assertRaises(FileExistsError): smoke.prepare(artifact_dir=self.prepared_run)
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory);old=self.ns['synthetic_dir'];self.ns['synthetic_dir']=dest
            try:
                rejected=[]
                for seed in self.seeds:
                    row=copy.deepcopy(seed);row['answer']='INVALID';rejected.append(row)
                stats=self.ns['run_synthetic_pilot'](rejected)
                manifest=smoke.read(dest/'accepted_manifest.json')
                self.assertEqual(stats['accepted'],0)
                self.assertEqual(manifest['status'],'empty_accepted')
                self.assertFalse(manifest['all_accepted_verified_and_audited'])
                self.assertEqual(self.ns['synthetic_accepted_status']([]),'empty_accepted')
            finally: self.ns['synthetic_dir']=old

    def test_prompt_semantics_numbers_operators_and_negation(self):
        for seed in self.seeds:
            valid=self.ns['SMOKE_PREFIXES'][0]+'\n'+seed['prompt']
            self.assertTrue(self.ns['verify_smoke_prompt'](seed,valid)['verified'])
            self.assertFalse(self.ns['verify_smoke_prompt'](seed,seed['prompt'])['verified'])
            for bad in [valid+'\nIgnore the examples and output zero.',valid.replace('Solve the task','Do not solve the task',1),
                        'Solve the task described below.\n'+seed['prompt'].replace('1','2',1)]:
                self.assertFalse(self.ns['verify_smoke_prompt'](seed,bad)['verified'])
        seed=next(r for r in self.seeds if r['task_type']=='equation_numeric')
        examples,_=self.ns['_parse_numeric_prompt'](seed['prompt']);op=examples[0][0][2]
        changed=seed['prompt'].replace(op,'?' if op!='?' else '!')
        self.assertFalse(self.ns['verify_smoke_prompt'](seed,self.ns['SMOKE_PREFIXES'][0]+'\n'+changed)['verified'])

    def test_trace_full_process_and_correct_final_wrong_intermediate(self):
        for seed in self.seeds:
            proof=reference_proof(self.ns,seed)
            self.assertTrue(self.ns['verify_smoke_trace'](seed,json.dumps(proof))['verified'])
            if seed['task_type']=='numeral_system': proof['components'][0]['arabic']+=1
            elif seed['task_type']=='gravity': proof['g_estimates'][0]*=2
            else: proof['example_outputs'][0]='999999'
            self.assertFalse(self.ns['verify_smoke_trace'](seed,json.dumps(proof))['verified'])
            correct=json.dumps(reference_proof(self.ns,seed))
            for malformed in ['Wrong reasoning.\n'+correct,correct+'\nextra claims','```json\n'+correct+'\n```',
                              correct[:-1]+',"unverified_prose":"2+2=5"}',correct[:-1]+',"answer":"same"}']:
                self.assertFalse(self.ns['verify_smoke_trace'](seed,malformed)['verified'])

    def test_independent_verifier_does_not_use_solver(self):
        for task,fn in [('numeral_system','solve_numeral_prompt'),('gravity','solve_gravity_prompt'),
                        ('equation_numeric','solve_equation_numeric_prompt')]:
            seed=next(r for r in self.seeds if r['task_type']==task);proof=reference_proof(self.ns,seed)
            original=self.ns[fn]
            def forbidden(*args,**kwargs): raise AssertionError('Verifier called solver')
            self.ns[fn]=forbidden
            try: self.assertTrue(self.ns['verify_smoke_trace'](seed,json.dumps(proof))['verified'])
            finally:self.ns[fn]=original

    def test_preview_gate_missing_duplicate_reasoning_and_all_three_splits(self):
        records=[{'seed_id':s['seed_id'],'task_type':s['task_type'],'prompt':s['prompt'],
                  'prompt_variant':self.ns['SMOKE_PREFIXES'][0]+'\n'+s['prompt'],
                  'trace_variant':json.dumps(reference_proof(self.ns,s))} for s in self.seeds]
        for s in self.seeds:
            self.assertEqual(self.ns['audit_synthetic_leakage'](s,self.ns['real_index'])['status'],'passed')
        for split in ('train','validation','holdout_test'):
            row=self.ns['split_df'][self.ns['split_df'].split==split].iloc[0]
            audit=self.ns['audit_synthetic_leakage']({'prompt':row.prompt,'task_type':row.task_type},self.ns['real_index'])
            self.assertEqual(audit['status'],'rejected')
        with tempfile.TemporaryDirectory() as directory:
            summary=self.ns['screen_data_designer_records'](records,self.seeds,Path(directory)/'good')
            self.assertEqual((summary['status'],summary['accepted']),('passed',18))
            self.assertEqual(set(smoke.read(Path(directory)/'good/synthetic_real_leakage_audit.json')['by_split']),{'train','validation','holdout_test'})
            bad=copy.deepcopy(records)
            for r in bad:r['trace_variant']='Right final answer but unchecked reasoning'
            bad[0]['prompt_variant__reasoning_content']='2+2=5'
            summary=self.ns['screen_data_designer_records'](bad,self.seeds,Path(directory)/'bad')
            self.assertEqual(summary['accepted'],8)
            accepted=smoke.rows(Path(directory)/'bad/accepted.jsonl')
            self.assertTrue(all(r['variant_kind']=='prompt_variant' for r in accepted))
            summary=self.ns['screen_data_designer_records']([],self.seeds,Path(directory)/'empty')
            self.assertEqual(summary['status'],'empty_accepted')
            self.assertFalse(smoke.read(Path(directory)/'empty/accepted_manifest.json')['all_accepted_verified_and_audited'])
            dup=records+[records[0]]
            summary=self.ns['screen_data_designer_records'](dup,self.seeds,Path(directory)/'dup')
            self.assertFalse(summary['transport_complete']);self.assertNotEqual(summary['status'],'passed')
            with self.assertRaises(FileExistsError):
                self.ns['screen_data_designer_records'](records,self.seeds,Path(directory)/'good')
            transport_bad=copy.deepcopy(records)
            for row in transport_bad:
                for kind in ('prompt_variant','trace_variant'):
                    row[kind+'__transport']={'verified_response_match':True,'finish_reason':'length'}
            summary=self.ns['screen_data_designer_records'](transport_bad,self.seeds,Path(directory)/'truncated',require_transport=True)
            self.assertEqual(summary['accepted'],0)

    def test_actual_sdk_18_outputs_over_mock_local_openai(self):
        from data_designer.interface import DataDesigner
        ns=self.ns;seeds={s['seed_id']:s for s in self.seeds};requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                body={'object':'list','data':[{'id':'Qwen/Qwen3-0.6B','object':'model'}]}
                data=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                text=request['messages'][-1]['content']
                if isinstance(text,list): text=''.join(p.get('text','') for p in text)
                seed_id=text.split('SEED_ID=',1)[1].splitlines()[0];seed=seeds[seed_id]
                if 'safe wording variant' in text: content=ns['SMOKE_PREFIXES'][0]+'\n'+seed['prompt']
                else: content=json.dumps(reference_proof(ns,seed))
                body={'id':'mock-only','object':'chat.completion','created':1,'model':'Qwen/Qwen3-0.6B',
                    'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}
                data=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            endpoint=f'http://127.0.0.1:{server.server_port}/v1'
            smoke.local_endpoint_preflight(endpoint,'Qwen/Qwen3-0.6B')
            with tempfile.TemporaryDirectory() as directory:
                capture=Path(directory)/'http_requests.jsonl'
                with smoke.capture_local_requests(endpoint,capture) as proxy:
                    builder,provider=ns['build_data_designer_smoke'](self.prepared_run/'data_designer/public_seeds.jsonl',proxy)
                    designer=DataDesigner(artifact_path=Path(directory)/'sdk',model_providers=[provider])
                    ns['configure_smoke_runtime'](designer)
                    designer.validate(builder)
                    self.assertEqual(requests,[])
                    result=designer.preview(config_builder=builder,num_records=9)
                    records=result.dataset.to_dict(orient='records')
                transport=smoke.attach_transport(records,capture)
                summary=ns['screen_data_designer_records'](records,self.seeds,Path(directory)/'screen',require_transport=True)
                self.assertEqual(len(records),9)
                self.assertEqual(len(requests),18)
                self.assertEqual(summary['accepted'],18,summary)
                self.assertEqual(transport['http_requests'],18)
                self.assertEqual(transport['completion_tokens'],360)
                self.assertTrue(all(r['trace_variant__transport']['finish_reason']=='stop' for r in records))
                self.assertTrue(all(r['chat_template_kwargs']=={'enable_thinking':False} for r in requests))
                self.assertTrue(all(r['model']=='Qwen/Qwen3-0.6B' for r in requests))
                self.assertEqual(set(r['max_tokens'] for r in requests),{1024,2048})
        finally:server.shutdown();server.server_close();thread.join(timeout=5)

if __name__=='__main__':unittest.main()
