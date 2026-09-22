"""Train on the training split; select weights using validation loss only."""
import time

PROCESS_STARTED = time.perf_counter()

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torchvision
from torch import nn

from config import (BATCH_SIZE, CPU_THREADS, EPOCHS, LEARNING_RATE,
                    SEED, TRAIN_SIZE)
from paths import OUTPUT_DIR
from data import training_loaders
from model import MLP


def measure(model, loader, loss_fn):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            loss_sum += loss_fn(scores, labels).item() * labels.size(0)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "correct": correct, "total": total}


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        scores = model(images)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": loss_sum / total, "accuracy": correct / total}


def train(train_size=TRAIN_SIZE, epochs=EPOCHS, seed=SEED, output_dir=OUTPUT_DIR,
          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, threads=CPU_THREADS):
    if epochs < 1 or batch_size < 1 or threads < 1 or learning_rate <= 0:
        raise ValueError("epoch・batch size・スレッド数・学習率は正の値で指定してください。")
    started = time.perf_counter()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    train_loader, validation_loader, split = training_loaders(train_size, batch_size, seed)
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "mnist_mlp.pt"
    print(f"CPU / 訓練{train_size:,}枚 / {epochs} epoch / seed {seed}", flush=True)
    print(f"保存先: {output_dir}（再実行すると同名の結果を更新します）", flush=True)
    history, best_loss, best_epoch = [], float("inf"), 0
    loop_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        training = train_epoch(model, train_loader, loss_fn, optimizer)
        validation = measure(model, validation_loader, loss_fn)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save(model.state_dict(), weights_path)
        row = {"epoch": epoch, "train": training, "validation": validation,
               "elapsed_seconds": time.perf_counter() - loop_started}
        history.append(row)
        print(f"epoch {epoch}/{epochs} | train loss {training['loss']:.4f} "
              f"acc {training['accuracy']:.2%} | val loss {validation['loss']:.4f} "
              f"acc {validation['accuracy']:.2%} | {row['elapsed_seconds']:.1f}秒", flush=True)
    run = {
        "model": "MLP 784-128-10 ReLU", "preprocessing": "float32 / 255; [N,1,28,28]",
        "train_size": train_size, "epochs": epochs, "seed": seed,
        "batch_size": batch_size, "optimizer": "Adam", "learning_rate": learning_rate,
        "saved_epoch": best_epoch, "validation": history[best_epoch - 1]["validation"],
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "loop_seconds": time.perf_counter() - loop_started,
        "train_seconds": time.perf_counter() - started,
        "environment": {"os": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "device": "cpu", "threads": threads},
        "split": split,
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存完了: {weights_path} / 採用epoch {best_epoch} / {run['train_seconds']:.1f}秒", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--threads", type=int, default=CPU_THREADS)
    args = parser.parse_args()
    try:
        train(**vars(args))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Python起動後の経過時間（importを含む）: {time.perf_counter() - PROCESS_STARTED:.1f}秒")


if __name__ == "__main__":
    main()
