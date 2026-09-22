"""Download once; use the same split and preprocessing in every execution."""
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from config import SPLIT_SEED, VALIDATION_SIZE, EVAL_BATCH_SIZE
from paths import DATA_DIR


def stratified_take(ids, labels, count, generator):
    """Proportional class counts, with deterministic largest-remainder rounding."""
    groups = [ids[labels[ids] == digit] for digit in range(10)]
    shares = torch.tensor([len(group) for group in groups], dtype=torch.float64)
    shares = shares * count / len(ids)
    quotas = shares.floor().to(torch.int64)
    order = sorted(range(10), key=lambda digit: (-(shares[digit] - quotas[digit]).item(), digit))
    for digit in order[:count - int(quotas.sum())]:
        quotas[digit] += 1
    selected, remaining = [], []
    for group, quota in zip(groups, quotas.tolist()):
        shuffled = group[torch.randperm(len(group), generator=generator)]
        selected.append(shuffled[:quota])
        remaining.append(shuffled[quota:])
    return torch.cat(selected).sort().values, torch.cat(remaining).sort().values


def split_indices(labels, train_size):
    if not 1 <= train_size <= len(labels) - VALIDATION_SIZE:
        raise ValueError(f"訓練枚数は1〜{len(labels) - VALIDATION_SIZE}で指定してください。")
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    validation, pool = stratified_take(torch.arange(len(labels)), labels, VALIDATION_SIZE, generator)
    train, _ = stratified_take(pool, labels, train_size, generator)
    return train, validation


def load_mnist(train):
    try:
        return MNIST(DATA_DIR, train=train, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            "MNISTがありません。教材ルートで uv run python 05-pytorch/data.py を実行してください。"
        ) from error


def tensor_dataset(dataset, ids):
    # Convert once, avoiding image decoding and conversion on every epoch.
    images = dataset.data[ids].unsqueeze(1).to(torch.float32).div_(255)
    return TensorDataset(images, dataset.targets[ids].to(torch.int64))


def training_loaders(train_size, batch_size, seed):
    dataset = load_mnist(train=True)
    train_ids, validation_ids = split_indices(dataset.targets, train_size)
    train = DataLoader(tensor_dataset(dataset, train_ids), batch_size=batch_size,
                       shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    validation = DataLoader(tensor_dataset(dataset, validation_ids),
                            batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    split = {
        "dataset": "MNIST official training split",
        "split_seed": SPLIT_SEED,
        "train_ids": train_ids.tolist(),
        "validation_ids": validation_ids.tolist(),
        "train_class_counts": torch.bincount(dataset.targets[train_ids], minlength=10).tolist(),
        "validation_class_counts": torch.bincount(dataset.targets[validation_ids], minlength=10).tolist(),
    }
    return train, validation, split


def test_loader():
    dataset = load_mnist(train=False)
    return DataLoader(tensor_dataset(dataset, torch.arange(len(dataset))),
                      batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)


def main():
    started = time.perf_counter()
    print(f"MNISTの保存先: {DATA_DIR}", flush=True)
    MNIST(DATA_DIR, train=True, download=True)
    MNIST(DATA_DIR, train=False, download=True)
    print(f"準備完了（{time.perf_counter() - started:.1f}秒）。学習時の再取得はありません。")


if __name__ == "__main__":
    main()
