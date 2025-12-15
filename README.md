# 수어 'ㄱ' 인식 AI (웹캠 + 손 랜드마크)

이 프로젝트는 **MediaPipe Hands 손 랜드마크(21점)** 를 추출해서 **‘ㄱ’(1) vs ‘그 외’(0)** 를 분류합니다.

## 구성

- `ksl_g/scripts/collect.py`: 웹캠으로 데이터 수집(랜드마크 → CSV)
- `ksl_g/scripts/train_model.py`: CSV로 모델 학습 → `joblib` 저장
- `ksl_g/scripts/predict_webcam.py`: 저장된 모델로 웹캠 실시간 인식

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ksl_g/requirements.txt
```

만약 `mediapipe` 설치가 실패하면, 파이썬 버전을 **3.10~3.11** 로 맞추는 게 가장 흔한 해결책입니다.

## 1) 데이터 수집

```bash
python ksl_g/scripts/collect.py --out ksl_g/data/landmarks.csv
```

수집 화면에서:

- `1`: 라벨을 **ㄱ(1)** 로 설정
- `0`: 라벨을 **other(0)** 로 설정
- `SPACE`: 현재 프레임의 랜드마크 1개 저장
- `q`: 종료

권장: 라벨별로 **최소 100개 이상** (가능하면 300~1000개).

## 2) 학습

```bash
python ksl_g/scripts/train_model.py --data ksl_g/data/landmarks.csv --out ksl_g/models/ksl_g.joblib
```

## 3) 실시간 인식(웹캠)

```bash
python ksl_g/scripts/predict_webcam.py --model ksl_g/models/ksl_g.joblib --threshold 0.7
```

## 팁(정확도 올리기)

- **배경/조명 다양화**: 밝은 곳/어두운 곳, 다른 배경에서 섞어서 수집
- **손 크기/거리 다양화**: 카메라와 거리 바꿔가며 수집
- **other(0)** 라벨 강화: ‘ㄱ’과 헷갈릴만한 손모양을 많이 넣을수록 오탐이 줄어듭니다