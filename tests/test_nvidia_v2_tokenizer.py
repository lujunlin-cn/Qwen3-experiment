"""Local tokenizer-only integration; no model, GPU, hosted API or downloads."""
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from utils import run_nvidia_v2 as v2


class V2TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.socket_guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        cls.gpu_guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA initialization forbidden'))
        cls.socket_guard.start()
        cls.gpu_guard.start()
        cls.addClassCleanup(cls.socket_guard.stop)
        cls.addClassCleanup(cls.gpu_guard.stop)
        cls.tokenizer, cls.metadata = v2.local_tokenizer()
        if cls.tokenizer is None: raise RuntimeError('Tokenizer integration requires a local Qwen3-0.6B tokenizer')
        cls.sft = v2.load_sft_render_definitions()
        cls.records = v2.rows(v2.SFT_PREVIEW / 'nvidia_sft_rendered_preview.jsonl')

    def test_actual_completion_and_sft_mask_token_counts(self):
        self.assertEqual(len(self.records), 6)
        for record in self.records:
            expected = record['actual_token_counts']
            self.assertEqual(expected['completion_tokens'], len(self.tokenizer(record['completion'], add_special_tokens=False).input_ids))
            encoded = self.sft['encode_chat_example'](self.tokenizer, record)
            self.assertEqual(expected['sft_full_chat_tokens'], len(encoded['input_ids']))
            self.assertTrue(all(x == -100 for x in encoded['labels'][:encoded['prompt_length']]))
            self.assertEqual(expected['sft_supervised_tokens'], sum(x != -100 for x in encoded['labels']))
            self.assertTrue(self.sft['audit_trace_record'](record)['accepted'])
            self.assertLessEqual(expected['sft_full_chat_tokens'], self.sft['max_length'])
            self.assertFalse(record['sft_training_eligible'])

    def test_repeat_export_is_byte_identical_without_network_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'repeat'
            result = v2.render(destination=destination, tokenizer_path=Path(self.metadata['path']))
            self.assertEqual(result['sft_format_audit_passed'], 6)
            name = 'nvidia_sft_rendered_preview.jsonl'
            self.assertEqual((destination / name).read_bytes(), (v2.SFT_PREVIEW / name).read_bytes())
            self.assertEqual(result['sft_training_eligible'], 0)
            self.assertEqual(result['finally_entered_sft'], 0)


if __name__ == '__main__':
    unittest.main()
