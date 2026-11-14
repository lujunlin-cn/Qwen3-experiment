"""CPU registration and frozen-input checks for the real GPU supervisor."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils import round2_run as runner
from utils.course_protocol import read,write,digest,check_protocol


class RunnerTests(unittest.TestCase):
    def test_independent_runs_preserve_panels_and_check_source_hashes(self):
        design=read(runner.ROOT/'configs/round2_design.json')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'model';model.mkdir();adapter=root/'adapter';adapter.mkdir()
            (model/'model.safetensors').write_bytes(b'cpu-registration-fixture')
            (adapter/'adapter_model.safetensors').write_bytes(b'cpu-adapter-fixture')
            write(model/'download_manifest.json',{'revision':design['model_revision'],
                'files':{'model.safetensors':{'sha256':digest(model/'model.safetensors')}}})
            original_digest=runner.digest
            def fake_digest(path):
                if Path(path)==adapter/'adapter_model.safetensors':return design['parent_sft_adapter_sha256']
                return original_digest(path)
            args=SimpleNamespace(suite=root/'suite',design=runner.ROOT/'configs/round2_design.json',
                model=model,adapter=adapter,inference_python=root/'inference/python',
                training_python=root/'training/python',nemo_python=root/'nemo/python',
                source_commit='fixture-not-gpu',budget_seconds=28800)
            with patch.object(runner,'digest',side_effect=fake_digest):runner.register(args)
            manifest=read(args.suite/'suite.json')
            self.assertEqual(len(set(manifest['runs'].values())),5)
            self.assertFalse(manifest['formal_training_started'])
            for method,path in manifest['runs'].items():
                protocol=check_protocol(path)
                self.assertFalse(protocol['holdout_used'])
                self.assertEqual(protocol['config']['dpo']['preference_mode'],'B')
                self.assertEqual(protocol['config']['ppo']['reward_ablation'],'R1')
                self.assertEqual(protocol['config']['grpo']['reward_ablation'],'R1')
                self.assertNotEqual(str(Path(path)),str(args.suite))
            with patch.object(runner,'digest',side_effect=fake_digest):
                with self.assertRaises(FileExistsError):runner.register(args)
            changed=Path(manifest['runs']['ppo'])/'panels/train.jsonl'
            changed.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen'):check_protocol(manifest['runs']['ppo'])


if __name__=='__main__':unittest.main()
