"""Check interpreter isolation without loading CUDA or launching vLLM."""
import json
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils import run_nemo_guarded as guarded


class GuardedSmokeTests(unittest.TestCase):
    def test_server_uses_virtual_environment_interpreter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = root / 'env'
            venv.EnvBuilder(symlinks=True).create(environment)
            python = environment / 'bin/python'
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps({
                'model_type': 'qwen3', 'hidden_size': 1024, 'num_hidden_layers': 28,
            }))
            output = root / 'output'
            output.mkdir()
            args = SimpleNamespace(config=guarded.DEFAULT_CONFIG, output=output,
                                   vllm_python=python, model=model)
            server = Mock()
            server.poll.return_value = 1
            with patch.object(guarded.subprocess, 'Popen', return_value=server) as launch:
                with self.assertRaisesRegex(RuntimeError, 'vLLM exited'):
                    guarded.worker(args)
                interpreter = launch.call_args.args[0][0]
            # Execute the selected interpreter to prove that its venv survives.
            prefix = subprocess.check_output([
                interpreter, '-c', 'import sys; print(sys.prefix)',
            ], text=True).strip()
            self.assertEqual(Path(prefix), environment)
            server.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
