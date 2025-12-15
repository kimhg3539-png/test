from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ksl_g.landmarks import extract_features_from_mediapipe_result, ensure_row_vector  # noqa: E402
from ksl_g.train import load_model  # noqa: E402


def put_text(img, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser(description="웹캠으로 KSL 'ㄱ' 실시간 인식")
    p.add_argument("--model", type=Path, default=Path("ksl_g/models/ksl_g.joblib"))
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.7, help="ㄱ 판정 임계값(확률)")
    p.add_argument("--smooth", type=int, default=7, help="확률 이동평균 윈도우 크기")
    args = p.parse_args()

    model = load_model(args.model)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. --cam 번호를 바꿔보세요.")
        return 2

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    probs = deque(maxlen=max(1, int(args.smooth)))

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            detected = False
            p_g = None
            if res.multi_hand_landmarks:
                detected = True
                hand_lms = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                feats = extract_features_from_mediapipe_result(hand_lms)
                X = ensure_row_vector(feats.features)
                proba = model.predict_proba(X)[0]
                # classes_ could be [0,1] but be safe
                idx_1 = int(np.where(model.named_steps["clf"].classes_ == 1)[0][0])
                p_g = float(proba[idx_1])
                probs.append(p_g)

            p_smooth = float(np.mean(probs)) if probs else 0.0
            is_g = detected and (p_smooth >= float(args.threshold))

            put_text(frame, "q=quit", 25)
            put_text(frame, f"model={args.model}", 55)
            put_text(frame, f"hand_detected={detected}", 85, (0, 255, 0) if detected else (0, 0, 255))
            put_text(frame, f"p(ㄱ)={p_g if p_g is not None else 'NA'}  smooth={p_smooth:.3f}", 115)
            put_text(
                frame,
                f"result={'ㄱ' if is_g else 'other'}  threshold={args.threshold}",
                145,
                (0, 255, 0) if is_g else (255, 255, 255),
            )

            cv2.imshow("predict - KSL ㄱ", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

