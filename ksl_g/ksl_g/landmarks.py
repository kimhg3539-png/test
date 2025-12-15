from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class HandFeatures:
    """21개 손 랜드마크를 (x,y,z)*21=63차원으로 정규화한 특징."""

    features: np.ndarray  # shape: (63,)
    handedness: Optional[str] = None  # "Left" | "Right" | None


def _rotate_2d(points_xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    r = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points_xy @ r.T


def normalize_hand_landmarks(
    xyz: np.ndarray,
    *,
    align_to_middle_mcp: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    MediaPipe Hands 랜드마크(21,3)를 카메라/거리 변화에 덜 민감하게 정규화.

    - **이동 불변**: 손목(0번)을 원점으로 이동
    - **스케일 불변**: 전체 포인트의 최대 거리로 나눔
    - **(선택) 회전 완화**: 손목(0) -> 중지 MCP(9) 방향을 기준으로 정렬
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.shape != (21, 3):
        raise ValueError(f"Expected xyz shape (21,3), got {xyz.shape}")

    # 1) translate: wrist to origin
    wrist = xyz[0].copy()
    xyz = xyz - wrist

    # 2) rotate (2D) so that wrist->middle_mcp aligns with +Y axis
    if align_to_middle_mcp:
        v = xyz[9, :2]  # (x,y)
        angle = float(np.arctan2(v[0], v[1] + eps))  # swap to align to +Y
        xyz_xy = _rotate_2d(xyz[:, :2], -angle)
        xyz = np.concatenate([xyz_xy, xyz[:, 2:3]], axis=1)

    # 3) scale
    d = np.linalg.norm(xyz, axis=1)
    scale = float(np.max(d))
    if scale < eps:
        scale = 1.0
    xyz = xyz / scale

    return xyz.reshape(-1)  # (63,)


def extract_features_from_mediapipe_result(
    hand_landmarks,
    handedness_label: Optional[str] = None,
) -> HandFeatures:
    """
    mediapipe 손 랜드마크 객체를 받아 HandFeatures로 변환.
    """
    xyz = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    feats = normalize_hand_landmarks(xyz)
    return HandFeatures(features=feats, handedness=handedness_label)


def ensure_row_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 2 and x.shape[0] == 1:
        return x
    raise ValueError(f"Expected (63,) or (1,63), got {x.shape}")

