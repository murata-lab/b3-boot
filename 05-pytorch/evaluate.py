"""Restore a fresh model and report accuracy on all 10,000 official test images."""
import time

PROCESS_STARTED = time.perf_counter()

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image

from config import CPU_THREADS
from paths import WEIGHTS_PATH
from data import test_loader
from model import MLP


def load_model(weights_path):
    if not Path(weights_path).is_file():
        raise FileNotFoundError(
            f"重みがありません: {weights_path}\n"
            "教材ルートで uv run python 05-pytorch/train.py を実行してください。"
        )
    model = MLP()
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(weights)
    except RuntimeError as error:
        raise ValueError("重みとMLPの構造が一致しません。同じmodel.pyで学習した重みを指定してください。") from error
    model.eval()
    return model


def image_record(image, label, prediction, index):
    buffer = io.BytesIO()
    pixels = image.squeeze().mul(255).round().to(torch.uint8).numpy()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return {"index": index, "label": int(label), "prediction": int(prediction),
            "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}


def evaluate(weights_path=WEIGHTS_PATH, output_path=None):
    started = time.perf_counter()
    torch.set_num_threads(CPU_THREADS)
    weights_path = Path(weights_path).resolve()
    model = load_model(weights_path)
    loader = test_loader()
    correct, total, loss_sum = 0, 0, 0.0
    examples, mistakes = [], []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            predictions = scores.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            loss_sum += loss_fn(scores, labels).item()
            for offset in range(len(labels)):
                if len(examples) < 8:
                    examples.append(image_record(images[offset], labels[offset], predictions[offset], total + offset))
                if len(mistakes) < 8 and predictions[offset] != labels[offset]:
                    mistakes.append(image_record(images[offset], labels[offset], predictions[offset], total + offset))
                if len(examples) == 8 and len(mistakes) == 8:
                    break
            total += labels.size(0)
    result = {"dataset": "MNIST official test split", "weights_path": str(weights_path),
              "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
              "accuracy": correct / total, "loss": loss_sum / total,
              "correct": correct, "total": total, "examples": examples, "mistakes": mistakes,
              "example_selection": "First 8 test IDs; first 8 misclassified test IDs in ascending order",
              "evaluation_seconds": time.perf_counter() - started}
    # A custom weights path gets its own adjacent result unless --output is set.
    output_path = Path(output_path) if output_path else weights_path.parent / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"読み込んだ重み: {weights_path}")
    print(f"Test accuracy: {correct / total:.2%} ({correct}/{total})")
    print(f"評価結果: {output_path} / {result['evaluation_seconds']:.1f}秒")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        evaluate(args.weights, args.output)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Python起動後の経過時間（importを含む）: {time.perf_counter() - PROCESS_STARTED:.1f}秒")


if __name__ == "__main__":
    main()
