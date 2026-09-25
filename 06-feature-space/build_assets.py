"""Build the small, fixed MNIST sample shown in chapter 06.

Run after completing 05:
  uv run --with scikit-learn --with umap-learn python 06-feature-space/build_assets.py
Only this build step needs scikit-learn and umap-learn. The lesson reads the JSON.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST
from umap import UMAP


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "05-pytorch"))
from model import MLP  # noqa: E402


def rounded(points):
    return np.round(points, 3).tolist()


def main():
    weights_path = ROOT / "outputs/05-pytorch/notebook/mnist_mlp.pt"
    if not weights_path.exists():
        raise SystemExit("05を実行し、outputs/05-pytorch/notebook/mnist_mlp.pt を保存してください。")

    dataset = MNIST(ROOT / "data", train=False, download=True)
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()
    rng = np.random.default_rng(6)
    chosen = np.concatenate([rng.choice(np.flatnonzero(labels == digit), 60, replace=False) for digit in range(10)])
    chosen.sort()
    sample = images[chosen]
    y = labels[chosen]
    x = sample.astype(np.float32).reshape(-1, 784) / 255

    torch.manual_seed(6)
    before = MLP().eval()
    after = MLP().eval()
    after.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        tensor = torch.from_numpy(x)
        hidden_before = before.relu(before.hidden(tensor)).numpy()
        hidden_after = after.relu(after.hidden(tensor)).numpy()
        accuracy = float((after(tensor).argmax(dim=1).numpy() == y).mean())

    # PCA is fitted to the trained features only. Each UMAP map is a separate fit.
    pca = PCA(n_components=2, svd_solver="full").fit(hidden_after)
    pca_points = pca.transform(hidden_after)
    umap_common = dict(n_components=2, random_state=6, n_jobs=1, init="spectral")
    default, wide = dict(n_neighbors=15, min_dist=0.1), dict(n_neighbors=60, min_dist=0.5)
    maps = {
        "pixels": UMAP(**default, **umap_common).fit_transform(x),
        "before": UMAP(**default, **umap_common).fit_transform(hidden_before),
        "after": UMAP(**default, **umap_common).fit_transform(hidden_after),
        "after_wide": UMAP(**wide, **umap_common).fit_transform(hidden_after),
    }

    def distances(a):
        norm = np.sum(a * a, axis=1)
        dist = norm[:, None] + norm[None, :] - 2 * a @ a.T
        np.fill_diagonal(dist, np.inf)
        return dist

    # Measured in the original space, not on the 2D maps.
    def neighbor_agreement(a, k=10):
        nearest_k = np.argsort(distances(a), axis=1)[:, :k]
        return round(float((y[nearest_k] == y[:, None]).mean()), 4)

    # A concrete example where the learned metric changes which image is closest.
    pixel_neighbor = distances(x).argmin(axis=1)
    feature_neighbor = distances(hidden_after).argmin(axis=1)
    candidates = np.flatnonzero((y[pixel_neighbor] != y) & (y[feature_neighbor] == y))
    anchor = int(candidates[0]) if len(candidates) else 0

    # 4-bit greyscale is enough to identify each handwriting sample in the tooltip.
    packed_images = []
    for row in sample:
        shades = (row.reshape(-1) // 17).astype(np.uint8)
        packed_images.append(bytes((shades[::2] << 4) | shades[1::2]).hex())
    data = {
        "source": "MNIST test set; 60 images per digit, random seed 6",
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "sample_count": len(chosen),
        "sample_accuracy": round(accuracy, 4),
        "neighbor_agreement": {"pixels": neighbor_agreement(x), "before": neighbor_agreement(hidden_before),
                               "after": neighbor_agreement(hidden_after)},
        "labels": y.tolist(),
        "images": packed_images,
        "pca": rounded(pca_points),
        "pca_variance": rounded(pca.explained_variance_ratio_),
        "maps": {key: rounded(value) for key, value in maps.items()},
        "umap_settings": {"default": default, "wide": wide},
        "neighbor_example": {"anchor": anchor, "pixel": int(pixel_neighbor[anchor]), "feature": int(feature_neighbor[anchor])},
        "example_hidden": rounded(hidden_after[anchor, :12]),
        "example_scores": rounded(after(torch.from_numpy(x[anchor:anchor + 1])).detach().numpy()[0]),
    }
    target = Path(__file__).resolve().parent / "assets" / "mnist_maps.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("wrote", target, target.stat().st_size, "bytes")
    print("sample accuracy", accuracy, "neighbor example labels", y[[anchor, pixel_neighbor[anchor], feature_neighbor[anchor]]])
    print("PCA variance", pca.explained_variance_ratio_)


if __name__ == "__main__":
    main()
