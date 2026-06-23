# QAT full-int8 experiment report

- 실험 시각: 2026-06-18 02:32 KST
- GPU 서버: `codex-isl-gpu1`, `CUDA_VISIBLE_DEVICES=0`, RTX 3090 1장
- Pi 경로: `[RASPBERRY_PI_PROJECT]`
- QAT 산출물: `generated_models/codex_ablation_20260617_1558/03_aug_geometric_brightness10.qat_full_int8_io.tflite`

## 목적

기존 `03_aug_geometric_brightness10.best.keras` 모델을 대상으로 QAT를 수행한 뒤 full-int8 I/O TFLite로 변환했다. 목표는 PTQ full-int8 대비 test accuracy와 Pi inference latency가 개선되는지 확인하는 것이다.

## 실행 메모

- TensorFlow/TFLite 변환 환경: `tensorflow/tensorflow:2.15.0-gpu`
- TFMOT 환경: `tensorflow-model-optimization==0.8.0`, `tf_keras==2.15.0`
- 중요 수정: Docker 이미지 안의 standalone `keras` 패키지가 TFMOT wrapper clone에서 `keras.src` layer를 만들어 실패했다. 따라서 QAT run에서는 `TF_USE_LEGACY_KERAS=1` 설정 후 standalone `keras`를 제거하고 `tf_keras`만 사용했다.
- QAT 방식: DenseNet121 전체 중 Conv2D/Dense 121개 layer만 annotate한 partial QAT. BatchNormalization은 TFMOT 기본 registry에서 직접 quantize하지 않았다.
- 학습 설정: epoch 8, batch 32, Adam lr `1e-5`, representative dataset 300장.
- split: train 2172장, val 272장, test 273장.

## 학습 결과

![](qat_full_int8_20260618_0232/figures/qat_training_curves.svg)

| 항목 | 값 |
|---|---:|
| best epoch | 8 |
| best val loss | 0.016737 |
| best val accuracy | 99.63% |
| float core test accuracy | 99.63% |
| initial QAT test accuracy | 99.63% |
| QAT Keras test accuracy | 99.63% |
| QAT TFLite test accuracy | 97.80% (267/273) |

QAT Keras 모델은 test 272/273으로 기존 float 계열과 같은 수준까지 유지됐다. 다만 full-int8 TFLite 변환 후에는 267/273으로 내려갔다.

## Accuracy 비교

| 모델 | 정확도 | 정답/전체 | 크기 |
|---|---:|---:|---:|
| Float/Dynamic/FP16 계열 | 99.63% | 272/273 | 모델별 상이 |
| PTQ full-int8 I/O | 98.17% | 268/273 | 7.055 MB |
| QAT full-int8 I/O | 97.80% | 267/273 | 7.097 MB |

QAT가 이번 설정에서는 PTQ보다 1장 더 틀렸다. confusion matrix 기준 QAT는 scissors에서 paper로 가는 오분류가 늘었다.

QAT TFLite confusion matrix:

```text
true\pred   scissors  rock  paper
scissors         86     1     4
rock              0    91     0
paper             1     0    90
```

## Pi pure invoke benchmark

![](qat_full_int8_20260618_0232/figures/qat_thread_invoke.svg)

| threads | invoke mean | total mean | delegate | 주요 op |
|---:|---:|---:|---:|---|
| 1 | 5.9925 ms | 6.0143 ms | 4 | CONV_2D 120, QUANTIZE 57 |
| 2 | 5.9306 ms | 5.9547 ms | 4 | CONV_2D 120, QUANTIZE 57 |
| 4 | 5.8550 ms | 5.8790 ms | 4 | CONV_2D 120, QUANTIZE 57 |

4-thread 기준 QAT invoke는 5.8550 ms이다. PTQ full-int8 대비 QAT 모델은 `QUANTIZE` op가 57개 추가되어 runtime 경로가 더 복잡하다.

## Pi camera pipeline timing

### No dashboard

![](qat_full_int8_20260618_0232/figures/qat_timing_no_dashboard_pie.svg)

| 항목 | mean | loop 비율 |
|---|---:|---:|
| total loop | 33.3356 ms | 100.00% |
| FPS | 29.998 | - |
| camera read | 24.5010 ms | 73.50% |
| preprocess total | 1.8106 ms | 5.43% |
| tflite invoke | 6.8624 ms | 20.59% |
| predict total | 6.9850 ms | 20.95% |

### Dashboard draw

![](qat_full_int8_20260618_0232/figures/qat_timing_dashboard_pie.svg)

| 항목 | mean | loop 비율 |
|---|---:|---:|
| total loop | 33.3292 ms | 100.00% |
| FPS | 30.004 | - |
| camera read | 11.6815 ms | 35.05% |
| dashboard draw | 13.7587 ms | 41.28% |
| preprocess total | 1.6330 ms | 4.90% |
| tflite invoke | 6.0906 ms | 18.27% |
| predict total | 6.2138 ms | 18.64% |

## PTQ full-int8와 QAT full-int8 비교

| 조건 | PTQ full-int8 | QAT full-int8 | 해석 |
|---|---:|---:|---|
| test accuracy | 98.17% | 97.80% | QAT가 1장 낮음 |
| dashboard invoke mean | 5.0216 ms | 6.0906 ms | QAT가 1.0690 ms 느림 (21.3%) |
| dashboard predict total | 5.1399 ms | 6.2138 ms | QAT가 더 느림 |
| no-dashboard invoke mean | 5.9006 ms | 6.8624 ms | QAT가 더 느림 |
| E2E FPS | 30.0 fps 근처 | 30.0 fps 근처 | camera cadence 때문에 전체 FPS는 동일 |

## 결론

이번 QAT ablation은 성공적으로 수행됐지만, 현재 설정에서는 채택할 이유가 약하다.

- accuracy: QAT full-int8는 267/273으로 PTQ full-int8 268/273보다 낮다.
- speed: QAT full-int8는 Pi에서 XNNPACK delegate 4개 구간을 타지만, 추가 `QUANTIZE` op 때문에 invoke가 PTQ보다 느리다.
- E2E FPS: 둘 다 30 fps 근처로 묶인다. 전체 병목은 camera cadence와 dashboard drawing이며, invoke만 보면 PTQ full-int8가 더 낫다.

따라서 현재 best runtime 후보는 계속 `03_aug_geometric_brightness10.full_int8_io.tflite`이고, QAT 모델은 ablation artifact로 보관하는 것이 맞다.

## 실행/백업

- Pi 실행: `./05_RUN_QAT_FULL_INT8_WINDOW.sh`
- Pi 모델: `generated_models/codex_ablation_20260617_1558/03_aug_geometric_brightness10.qat_full_int8_io.tflite`
- 로컬 artifact: `[LOCAL_WORKSPACE]\qat_full_int8_20260618_0232`
- GPU artifact: `~[TRAINING_WORKSPACE]/rps_ablation_20260617_1558/results/04_qat_conv_dense_e8_lr1e-5_tfkeras_nokeras`

