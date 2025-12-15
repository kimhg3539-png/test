from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ksl_g.io import read_samples_csv  # noqa: E402
from ksl_g.train import save_model, train_binary_classifier  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="KSL 'ㄱ' 이진 분류기 학습")
    p.add_argument("--data", type=Path, default=Path("ksl_g/data/landmarks.csv"))
    p.add_argument("--out", type=Path, default=Path("ksl_g/models/ksl_g.joblib"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X, y = read_samples_csv(args.data)
    n = int(X.shape[0])
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if n < 20 or pos < 5 or neg < 5:
        print(
            "데이터가 너무 적습니다. 최소한 라벨별로 30~100개 이상 수집하는 걸 권장해요.\n"
            f"현재 N={n}, ㄱ(1)={pos}, other(0)={neg}"
        )

    res = train_binary_classifier(X, y, test_size=args.test_size, random_state=args.seed)
    save_model(args.out, res.model)

    print(f"Saved model -> {args.out}")
    print(f"Accuracy={res.accuracy:.4f}")
    print(res.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

