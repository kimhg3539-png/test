from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


FEATURE_DIM = 63


@dataclass(frozen=True)
class SampleRow:
    y: int  # 1: ㄱ, 0: other
    x: np.ndarray  # shape: (63,)
    handedness: Optional[str] = None


def write_samples_csv(path: Path, rows: Iterable[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["y"] + [f"f{i}" for i in range(FEATURE_DIM)] + ["handedness"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            x = np.asarray(r.x, dtype=np.float32).reshape(-1)
            if x.shape[0] != FEATURE_DIM:
                raise ValueError(f"Expected x dim {FEATURE_DIM}, got {x.shape}")
            w.writerow([int(r.y)] + [float(v) for v in x.tolist()] + [r.handedness or ""])


def append_sample_csv(path: Path, row: SampleRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["y"] + [f"f{i}" for i in range(FEATURE_DIM)] + ["handedness"])
        x = np.asarray(row.x, dtype=np.float32).reshape(-1)
        if x.shape[0] != FEATURE_DIM:
            raise ValueError(f"Expected x dim {FEATURE_DIM}, got {x.shape}")
        w.writerow([int(row.y)] + [float(v) for v in x.tolist()] + [row.handedness or ""])


def read_samples_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N,63), y: (N,)
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    xs: List[List[float]] = []
    ys: List[int] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y = int(row["y"])
            x = [float(row[f"f{i}"]) for i in range(FEATURE_DIM)]
            ys.append(y)
            xs.append(x)
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y

