# Strong augmentation top-2 실험 보고서

- 실험 시작: 2026-06-18 17:52 KST
- 실행 위치: `codex-isl-gpu1`, GPU 0, TensorFlow 2.15 Docker
- 목적: DenseNet121은 유지하고 train augmentation을 강화해서 현재 top-2 변환 recipe의 정확도 개선 여부 확인
- 기준 best: `03_aug_geometric_brightness10`
- 입력 크기: 64x64x3
- split: train 2172장, val 272장, test 273장

## TL;DR

강한 augmentation은 효과가 있었다. 최종 추천 모델은 `strong PTQ random_c120_s31`이다.

기존 best PTQ/QAT는 test 272/273 = 99.63%였는데, strong augmentation으로 재학습한 base에서 PTQ `random_c120_s31`을 다시 적용하니 test 273/273 = 100.00%가 나왔다. Pi latency도 4-thread invoke 4.79 ms로 기존 best와 같은 속도권이다.

QAT conv5/head는 Keras QAT model 자체는 test 273/273이었지만, TFLite full-int8 변환 후에는 272/273으로 남았다. 따라서 배포용 최종 후보는 QAT가 아니라 PTQ 쪽이다.

## Augmentation 설정

학습용 모델에만 아래 augmentation을 적용했다. 배포/변환용 `.keras`는 augmentation layer를 제거한 deploy model로 따로 저장했다.

- `RandomFlip(horizontal)`
- `RandomRotation(0.12)`
- `RandomTranslation(0.12, 0.12)`
- `RandomZoom(0.15)`
- `RandomContrast(0.25)`
- `RandomBrightness(0.25, value_range=(0, 255))`
- `GaussianNoise(3.0)`

기존 best weight에서 DenseNet121과 Dense head weight를 transfer한 뒤 learning rate `5e-6`로 fine-tuning했다.

## Base 학습 결과

![](strong_aug_top2_20260618_1752/figures/strong_aug_training_qat_curves.svg)

| 항목 | 결과 |
|---|---:|
| epochs requested | 30 |
| epochs ran | 14 |
| selected epoch | 2 |
| selection | max val accuracy, tie min val loss |
| val accuracy | 272/272 = 100.00% |
| test accuracy | 273/273 = 100.00% |

기존 base는 test 272/273이었고, strong augmentation 후 deploy Keras base는 test 273/273으로 올라갔다.

## Top-2 변환 결과

![](strong_aug_top2_20260618_1752/figures/strong_aug_accuracy_latency.svg)

| 모델 | 변환 | val accuracy | test accuracy | Pi 4-thread invoke mean | 판단 |
|---|---|---:|---:|---:|---|
| old best PTQ | PTQ random c120 s31 | 기존 | 272/273 = 99.63% | 4.89 ms | 이전 best |
| old best QAT | QAT conv5/head | 기존 | 272/273 = 99.63% | 4.84 ms | 이전 best |
| strong PTQ random | PTQ random c120 s31 | 270/272 = 99.26% | 273/273 = 100.00% | 4.79 ms | 최종 추천 |
| strong PTQ brightness | PTQ brightness mild c120 s17 | 270/272 = 99.26% | 273/273 = 100.00% | 4.79 ms | 후보 |
| strong QAT conv5/head | QAT conv5/head e18 | 271/272 = 99.63% | 272/273 = 99.63% | 4.82 ms | 개선 없음 |

## QAT 관찰

QAT는 18 epoch 동안 매 epoch TFLite 변환 후 validation을 수행했다.

- best epoch by TFLite validation: 15
- best TFLite val: 271/272 = 99.63%
- Keras QAT test: 273/273 = 100.00%
- TFLite full-int8 test: 272/273 = 99.63%

즉 학습 모델 자체는 100%까지 올라갔지만, full-int8 TFLite 변환 후 한 장이 다시 틀렸다. QAT 경로의 남은 문제는 training accuracy가 아니라 TFLite quantized graph의 표현/스케일링 쪽으로 보인다.

## 파일

새 best 백업:

- `[LOCAL_WORKSPACE]\best_models_backup_20260618_1833_strong_aug`

배포 후보:

| 파일 | SHA256 |
|---|---|
| `04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite` | `C34199564DE4F7ADCFBA836D9A33BAE472739ACA8F1711C63E2E6335950FBBC7` |
| `04_aug_strong_flip_light25.ptq_brightness_mild_c120_s17_full_int8_io.tflite` | `A869C7F356D1CA0F67D12F81DEF3C6557523E4D859F50203FF0DE0DBD239191F` |
| `04_aug_strong_flip_light25.qat_conv5_head_e18_full_int8_io.tflite` | `08C18EC4A44FD46F385128943A513AD4F65A04BFC387F41308A7A0C965C39C5F` |

## 결론

현재 최종 배포 모델은 `04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite`로 업데이트하는 것이 좋다.

이 모델은 DenseNet121 구조를 유지했고, 기존 best와 같은 full-int8 PTQ 경로를 사용하며, Pi latency도 기존과 같은 4.8 ms 수준이다. test set 기준으로는 272/273에서 273/273으로 개선됐다.


