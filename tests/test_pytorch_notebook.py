"""Run the actual teaching cells offline, including saving and reloading weights."""
import contextlib
import ast
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "05-pytorch" / "notebook.py"


class SyntheticMNIST:
    """Same public data interface and sizes; deterministic images, no network."""
    calls = []

    def __init__(self, root, train, download):
        self.calls.append(train)
        count = 60000 if train else 10000
        self.targets = torch.arange(count) % 10
        self.data = (self.targets * 25).to(torch.uint8).view(-1, 1, 1).expand(-1, 28, 28)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.fromarray(self.data[index].numpy()), int(self.targets[index])


class NotebookWorkflow(unittest.TestCase):
    def test_complete_training_example_needs_no_earlier_cell_variables(self):
        tree = ast.parse(NOTEBOOK.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_mnist")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        original_cwd = Path.cwd()
        SyntheticMNIST.calls = []
        import sys
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "path", [str(NOTEBOOK.parent), *sys.path]):
            try:
                os.chdir(directory)
                with patch("torchvision.datasets.MNIST", SyntheticMNIST), contextlib.redirect_stdout(io.StringIO()):
                    model, weights, epoch, history = namespace["train_mnist"](epochs=1)
                self.assertEqual(SyntheticMNIST.calls, [True])
                self.assertEqual(epoch, 1)
                self.assertEqual(len(history), 1)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(weights[name], value)
            finally:
                os.chdir(original_cwd)

    def test_actual_cells_train_save_restore_and_evaluate_offline(self):
        spec = importlib.util.spec_from_file_location("pytorch_teaching_notebook", NOTEBOOK)
        notebook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(notebook)
        SyntheticMNIST.calls = []
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch("torchvision.datasets.MNIST", SyntheticMNIST), contextlib.redirect_stdout(io.StringIO()):
                    _, values = notebook.app.run()
                self.assertEqual(SyntheticMNIST.calls, [True, True, False])
                self.assertEqual(len(values["train_data"]), 40000)
                self.assertEqual(len(values["validation_data"]), 5000)
                self.assertFalse(set(values["train_ids"].tolist()) & set(values["validation_ids"].tolist()))
                history = values["history"]
                self.assertEqual(len(history), 5)
                self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
                self.assertEqual(values["best_epoch"], min(history, key=lambda row: row["loss"])["epoch"])
                saved = torch.load(values["weights_path"], weights_only=True)
                restored = values["restored_model"]
                for name, tensor in values["best_weights"].items():
                    torch.testing.assert_close(saved[name], tensor)
                    torch.testing.assert_close(restored.state_dict()[name], tensor)
                self.assertFalse(restored.training)
                self.assertIsNot(restored, values["trained_model"])
                # Metrics displayed by the test cell must describe the restored weights.
                before = {k: v.clone() for k, v in restored.state_dict().items()}
                metrics = values["evaluate"](restored, values["test_loader"], values["loss_fn"])
                self.assertEqual(metrics, values["test_result"])
                self.assertTrue(all(torch.equal(before[k], v) for k, v in restored.state_dict().items()))
            finally:
                os.chdir(original_cwd)
                import matplotlib.pyplot as plt
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
