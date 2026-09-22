"""Offline checks for the chapter's data, saved models, and displayed source."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1] / "05-pytorch"
sys.path.insert(0, str(CHAPTER))
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from paths import OUTPUT_DIR
from data import split_indices
from evaluate import load_model
from lesson_ui import file_preview, load_results
from model import MLP
from train import measure, train_epoch


class PytorchLesson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_split_is_reproducible_disjoint_and_proportional(self):
        labels = torch.arange(60000) % 10
        train, validation = split_indices(labels, 40000)
        again, same_validation = split_indices(labels, 40000)
        self.assertTrue(torch.equal(train, again))
        self.assertTrue(torch.equal(validation, same_validation))
        self.assertEqual(len(train), 40000)
        self.assertEqual(len(validation), 5000)
        self.assertEqual(len(set(train.tolist()) & set(validation.tolist())), 0)
        self.assertEqual(torch.bincount(labels[train]).tolist(), [4000] * 10)
        _, validation_other_size = split_indices(labels, 20000)
        self.assertTrue(torch.equal(validation, validation_other_size))

    def test_metrics_weight_the_short_final_batch(self):
        scores = torch.tensor([[3., 1.], [2., 4.], [4., 2.]])
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(scores, labels), batch_size=2)
        result = measure(nn.Identity(), loader, nn.CrossEntropyLoss())
        self.assertEqual((result['correct'], result['total']), (2, 3))
        self.assertAlmostEqual(result['loss'], nn.CrossEntropyLoss()(scores, labels).item(), places=6)

    def test_training_updates_weights_but_evaluation_does_not(self):
        torch.manual_seed(17)
        model = MLP()
        loader = DataLoader(TensorDataset(torch.rand(5, 1, 28, 28), torch.arange(5)), batch_size=3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        train_epoch(model, loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=.001))
        after = {k: v.clone() for k, v in model.state_dict().items()}
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in after.items()))
        measure(model, loader, nn.CrossEntropyLoss())
        self.assertTrue(all(torch.equal(after[k], v) for k, v in model.state_dict().items()))
        self.assertFalse(model.training)

    def test_saved_model_restores_scores_in_a_separate_process(self):
        torch.manual_seed(11)
        model = MLP().eval()
        inputs = torch.rand(3, 1, 28, 28)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            torch.save(model.state_dict(), folder / 'weights.pt')
            torch.save(inputs, folder / 'input.pt')
            script = ('import sys, torch; from pathlib import Path; '
                      'from evaluate import load_model; '
                      'p=Path(sys.argv[1]); '
                      'm=load_model(p/"weights.pt"); '
                      'torch.save(m(torch.load(p/"input.pt", weights_only=True)).detach(), p/"scores.pt")')
            subprocess.run([sys.executable, '-c', script, str(folder)], cwd=CHAPTER, check=True)
            actual = torch.load(folder / 'scores.pt', weights_only=True)
            torch.testing.assert_close(actual, model(inputs))
            wrong = folder / 'wrong.pt'
            torch.save(nn.Linear(2, 3).state_dict(), wrong)
            with self.assertRaisesRegex(ValueError, '構造が一致'):
                load_model(wrong)
            with self.assertRaisesRegex(FileNotFoundError, 'train.py'):
                load_model(folder / 'missing.pt')

    def test_import_has_no_training_side_effect_and_paths_are_stable(self):
        script = ('import sys; from pathlib import Path; '
                  f'sys.path.insert(0, {str(CHAPTER)!r}); '
                  'import model, train, evaluate, data; from paths import OUTPUT_DIR; '
                  'print(OUTPUT_DIR)')
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-c', script], cwd=folder, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), str(OUTPUT_DIR))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_file_view_is_actual_source_without_code_blocks(self):
        class Lines(HTMLParser):
            def __init__(self):
                super().__init__()
                self.depth = 0
                self.lines = []
            def handle_starttag(self, tag, attrs):
                if self.depth:
                    self.depth += 1
                elif ('class', 'pt-text') in attrs:
                    self.depth = 1
                    self.lines.append('')
            def handle_endtag(self, tag):
                if self.depth:
                    self.depth -= 1
            def handle_data(self, data):
                if self.depth:
                    self.lines[-1] += data
        html = file_preview(CHAPTER / 'model.py')
        self.assertNotIn('<pre', html)
        self.assertNotIn('<code', html)
        parsed = Lines()
        parsed.feed(html)
        self.assertEqual([line.rstrip() for line in parsed.lines], (CHAPTER / 'model.py').read_text().splitlines())

    def test_old_evaluation_cannot_be_shown_for_new_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            weights = b'new weights'
            (folder / 'mnist_mlp.pt').write_bytes(weights)
            (folder / 'run.json').write_text(json.dumps({'weights_sha256': hashlib.sha256(weights).hexdigest()}))
            (folder / 'history.json').write_text('[]')
            (folder / 'evaluation.json').write_text(json.dumps({'weights_sha256': 'old'}))
            self.assertIsNone(load_results(folder)['evaluation'])
            (folder / 'mnist_mlp.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '一致しません'):
                load_results(folder)


if __name__ == '__main__':
    unittest.main()
