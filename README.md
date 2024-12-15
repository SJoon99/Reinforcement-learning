# 강화학습 프로젝트 ( 2024_2 )

## PPO Atari LSTM 

PPO+LSTM 구조에서 CNN 구조 개선과 보상 예측 모듈 도입을 통해 학습 효율성과 안정성을 향상시키고자 함

## 구현 

- **EnhancedCNN**: 향상된 CNN 아키텍처를 사용한 PPO LSTM
- **RP_1step**: 1-step reward prediction을 사용한 PPO LSTM 
- **RP_3step**: 3-step reward prediction을 사용한 PPO LSTM 
- **RP_5step**: 5-step reward prediction을 사용한 PPO LSTM

## 실행 방법

### EnhancedCNN 버전
```bash
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari_lstm_EnhancedCNN.py --track --capture_video" \
    --num-seeds 5 \
    --workers 5
```

### RP_1step 버전 
```bash
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari_lstm_RP_1step.py --track --capture_video" \
    --num-seeds 5 \
    --workers 5
```

### RP_3step 버전 
```bash
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari_lstm_RP_3step.py --track --capture_video" \
    --num-seeds 5 \
    --workers 5
```

### RP_5step 버전 
```bash
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari_lstm_RP_5step.py --track --capture_video" \
    --num-seeds 5 \
    --workers 5
```


### 파라미터 설정
- **reward_pred_coef**: 보상 예측 손실의 가중치를 조절하는 계수
  - 값이 클수록 (예: > 0.5) 보상 예측에 더 집중
  - 값이 작을수록 (예: < 0.5) 정책 학습에 더 집중
  - 기본값은 0.5로 설정되어 있으며, 필요에 따라 조정

- **reward_prediction_horizon**: 몇 스텝 앞의 보상까지 예측할지 결정하는 파라미터
  - RP_1step, RP_3step, RP_5step 버전에 따라 각각 1, 3, 5로 설정
  - 값이 클수록 더 먼 미래의 보상을 예측하려 시도
  
