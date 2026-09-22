"""Stable defaults for the optional CLI; the notebook specifies its own paths."""
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CHAPTER_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "05-pytorch"
WEIGHTS_PATH = OUTPUT_DIR / "mnist_mlp.pt"
