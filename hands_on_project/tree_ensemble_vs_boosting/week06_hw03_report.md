# Week 06 HW03 - Tree Ensemble vs Boosting 비교

## 데이터셋 소개

Kaggle Titanic 데이터셋(train.csv, 891명)을 사용했다. 승객의 성별, 나이, 요금 등의 피처로 생존 여부를 예측하는 이진 분류 문제다.

## 실험 구조

- 데이터 분할: train/val/test = 60/20/20 (stratified, random_state=42)
- 평가 지표: Accuracy, F1 Score
- 비교 모델: RandomForest, XGBoost, LightGBM (기본), LightGBM (튜닝), LightGBM (튜닝+Early Stopping)
- LightGBM 튜닝 파라미터: num_leaves (15, 31, 50), learning_rate (0.01, 0.1, 0.3)

## 모델 비교표

| 모델 | Val Acc | Val F1 | Test Acc | Test F1 |
|------|---------|--------|----------|---------|
| RandomForest | 0.8258 | 0.7559 | 0.8156 | 0.7402 |
| XGBoost | 0.8146 | 0.7402 | 0.7933 | 0.7087 |
| LightGBM (기본) | 0.8427 | 0.7846 | 0.8045 | 0.7200 |
| LightGBM (튜닝) | 0.8539 | 0.7869 | 0.8212 | 0.7460 |
| LightGBM (튜닝+ES) | 0.8596 | 0.7967 | 0.8156 | 0.7273 |

## 분석

### Random Forest와 Boosting의 차이

Random Forest는 독립적인 트리를 병렬로 만들어 평균내는 방식이고, Boosting(XGBoost, LightGBM)은 이전 트리의 오차를 순차적으로 줄여나간다. 흥미롭게도 기본 파라미터에서는 RF가 XGBoost보다 나은 결과를 보였다. LightGBM은 튜닝 후에야 RF를 넘었는데, Boosting이 항상 우월한 건 아니고 데이터셋과 파라미터에 따라 달라진다는 걸 느꼈다.

### XGBoost vs LightGBM

둘 다 기본 파라미터로 비슷한 성능이 나왔다. LightGBM이 학습 속도가 체감상 더 빨랐고, verbosity 옵션으로 로그 제어가 편했다. XGBoost는 eval_metric을 명시적으로 지정해야 warning이 안 떠서 약간 번거로웠다.

### Early Stopping 효과

n_estimators를 500으로 넉넉하게 잡고 early_stopping(20)을 적용했을 때, 38번째 iteration에서 학습이 조기 종료되었다. val F1이 0.7869에서 0.7967로 소폭 개선되었고, 불필요한 학습을 줄이면서도 성능이 유지되었다. 과적합 방지에 효과적인 전략이다.

### 최종 모델 선택

튜닝된 LightGBM + Early Stopping 조합을 최종 모델로 선택했다. 하이퍼파라미터 튜닝으로 기본 대비 성능이 개선되었고, Early Stopping으로 학습 효율까지 확보할 수 있었다.

## Acknowledgment

본 과제의 코드 초안은 Claude Code를 활용해 작성했으며, 작성자가 직접 검수하고 로컬 환경에서 실행해 결과를 확인하였다.
