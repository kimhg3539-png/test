from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
from typing import Optional

import cv2
import mediapipe as mp

# Allow running without installation: python ksl_g/scripts/collect.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ksl_g.io import SampleRow, append_sample_csv  # noqa: E402
from ksl_g.landmarks import extract_features_from_mediapipe_result  # noqa: E402


def put_text(img, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser(description="KSL 'ㄱ' 데이터(손 랜드마크) 수집")
    p.add_argument("--out", type=Path, default=Path("ksl_g/data/landmarks.csv"))
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--max-hands", type=int, default=1)
    args = p.parse_args()

    label: Optional[int] = None  # 1: ㄱ, 0: other
    saved = 0
    last_msgs = deque(maxlen=3)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. --cam 번호를 바꿔보세요.")
        return 2

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=args.max_hands,
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
            if res.multi_hand_landmarks:
                detected = True
                for i, hand_lms in enumerate(res.multi_hand_landmarks):
                    handed = None
                    if res.multi_handedness and i < len(res.multi_handedness):
                        handed = res.multi_handedness[i].classification[0].label
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            put_text(frame, "Keys: 1=ㄱ, 0=other, SPACE=save, q=quit", 25)
            put_text(frame, f"label={label}  saved={saved}  out={args.out}", 55)
            put_text(frame, f"hand_detected={detected}", 85, (0, 255, 0) if detected else (0, 0, 255))
            y = 115
            for m in list(last_msgs)[-3:]:
                put_text(frame, m, y, (255, 255, 0))
                y += 25

            cv2.imshow("collect - KSL ㄱ", frame)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                break
            if k == ord("1"):
                label = 1
                last_msgs.appendleft("라벨을 1(ㄱ)로 설정")
            if k == ord("0"):
                label = 0
                last_msgs.appendleft("라벨을 0(other)로 설정")

            if k == 32:  # SPACE
                if label is None:
                    last_msgs.appendleft("먼저 라벨(1 또는 0)을 선택하세요.")
                    continue
                if not res.multi_hand_landmarks:
                    last_msgs.appendleft("손이 감지되지 않아 저장하지 않았습니다.")
                    continue
                # 1손만 사용(가장 첫 손)
                hand_lms = res.multi_hand_landmarks[0]
                handed = None
                if res.multi_handedness:
                    handed = res.multi_handedness[0].classification[0].label
                feats = extract_features_from_mediapipe_result(hand_lms, handedness_label=handed)
                append_sample_csv(args.out, SampleRow(y=label, x=feats.features, handedness=feats.handedness))
                saved += 1
                last_msgs.appendleft(f"저장 완료: y={label}, handedness={handed or ''}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

