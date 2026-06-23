# PTQ 정확도 회복 / QAT 경량화 실험 보고서

- 실험 시작 기준: 2026-06-18 15:28 KST
- dataset split: train 2172, val 272, test 273, seed 123
- 목표: PTQ 정확도를 올리거나, QAT 모델의 Pi inference time을 줄일 수 있는지 확인

## 결론

이번 실험에서 가장 좋은 방향은 두 개다.

1. PTQ focused calibration: `random_c120_s31`이 test 272/273 (99.63%), Pi 4-thread invoke 4.850 ms를 달성했다.
2. QAT scope 축소: `conv5_head_e18`이 test 272/273 (99.63%), Pi 4-thread invoke 4.850 ms를 달성했다.

기존 full QAT는 test 정확도는 같지만 `QUANTIZE` op가 57개이고 4-thread invoke가 약 5.84 ms였다. 이번 두 후보는 약 4.85~4.89 ms라 inference만 보면 16~17% 빨라졌다.

## 핵심 비교

![](ptq_qat_scope_20260618_1528/figures/invoke_compare.svg)

| 모델 | test accuracy | val best | annotated/Q ops | Pi 4T invoke | 해석 |
|---|---:|---:|---:|---:|---|
| 기존 full QAT rep600 e30 | 272/273 (99.63%) | 272/272 | annotated 121, QUANTIZE 57 | 5.827 ms | 기존 best |
| PTQ random c120 s31 | 272/273 (99.63%) | 271/272 (99.63%) | annotated 0, QUANTIZE 0 | 4.850 ms | 가장 단순하고 빠름 |
| QAT conv5_head e18 | 272/273 (99.63%) | 271/272 | annotated 33, QUANTIZE 15 | 4.850 ms | QAT 유지하면서 빠름 |
| QAT head e18 | 186/273 (68.13%) | 187/272 | annotated 1 | n/a | QAT 범위가 너무 좁아 실패 |

## PTQ focused calibration

![](ptq_qat_scope_20260618_1528/figures/ptq_top10_test_accuracy.svg)

| rank | variant | mode | rep | val | test |
|---:|---|---|---:|---:|---:|
| 1 | random_c120_s31 | random | 120 | 271/272 (99.63%) | 272/273 (99.63%) |
| 2 | random_c360_s31 | random | 360 | 270/272 (99.26%) | 271/273 (99.27%) |
| 3 | balanced_c60 | balanced | 60 | 270/272 (99.26%) | 270/273 (98.90%) |
| 4 | random_c180_s17 | random | 180 | 268/272 (98.53%) | 270/273 (98.90%) |
| 5 | balanced_c360 | balanced | 360 | 266/272 (97.79%) | 270/273 (98.90%) |
| 6 | random_c60_s17 | random | 60 | 269/272 (98.90%) | 269/273 (98.53%) |
| 7 | brightness_mild_c300_s17 | brightness_mild | 300 | 267/272 (98.16%) | 269/273 (98.53%) |
| 8 | random_c180_s31 | random | 180 | 269/272 (98.90%) | 268/273 (98.17%) |
| 9 | brightness_mild_c180_s17 | brightness_mild | 180 | 269/272 (98.90%) | 268/273 (98.17%) |
| 10 | random_c60_s31 | random | 60 | 268/272 (98.53%) | 268/273 (98.17%) |

이전에는 300/600/1000/2172장 위주로 보면서 대표 데이터 수를 늘렸는데, 이번에는 작은 subset과 seed를 흔들었다. 결과적으로 `120장 random subset` 하나가 full QAT와 같은 test 정확도를 냈다. 즉 PTQ는 데이터 양보다 calibration subset의 activation range가 훨씬 중요했다.

## QAT scope 실험

| scope | annotated layers | best val | test | Pi op 변화 |
|---|---:|---:|---:|---|
| head | 1 | 187/272 | 186/273 (68.13%) | 너무 적게 QAT해서 full-int8 오차 미보정 |
| conv5_head | 33 | 271/272 | 272/273 (99.63%) | QUANTIZE 57 -> 15, invoke 5.84 -> 4.85 ms |

`head only`는 Keras validation accuracy가 좋아도 TFLite-val이 무너졌다. 반대로 `conv5_head`는 마지막 Dense block과 head만 QAT해도 정확도가 유지되면서 graph가 꽤 가벼워졌다.

## E2E timing

| 모델 | total loop | predict total | tflite invoke | camera | dashboard | FPS |
|---|---:|---:|---:|---:|---:|---:|
| PTQ c120 | 33.33 ms | 5.09 ms | 4.98 ms | 12.98 ms | 13.59 ms | 30.00 |
| QAT conv5_head | 33.33 ms | 5.11 ms | 4.99 ms | 12.08 ms | 14.44 ms | 30.00 |

모델 invoke는 줄었지만 전체 FPS는 여전히 30 FPS cap 근처다. 카메라 read와 dashboard draw가 loop 시간을 채우기 때문이다. 그래도 model path만 보면 이번 PTQ/QAT conv5 후보가 기존 full QAT보다 확실히 낫다.

## 추천

1. 당장 배포 후보는 `PTQ random_c120_s31`과 `QAT conv5_head_e18` 둘 다 백업해 둔다.
2. 실제 카메라 손동작에서 안정성을 우선하면 QAT conv5_head를 먼저 써본다. calibration seed 운에 덜 의존할 가능성이 있다.
3. 속도와 단순성을 우선하면 PTQ random_c120_s31을 쓴다. graph에 extra `QUANTIZE`가 없고 invoke가 가장 작다.
4. 다음 실험은 PTQ random subset을 120장 근처에서 20 seed 정도 더 돌려서 robust한 calibration subset을 고르는 것이 좋다.

