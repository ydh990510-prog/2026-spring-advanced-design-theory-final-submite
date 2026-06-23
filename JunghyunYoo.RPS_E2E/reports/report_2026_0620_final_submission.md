# RPS On-Device AI 최종 결과 보고서

- 작성 기준: 2026-06-20 KST
- 범위: `report_2026_0615_1258_training.md` 이후 진행한 quantization, Raspberry Pi 실시간 pipeline, 카메라, latency 최적화 결과 정리
- 최종 목표: DenseNet121 기반 RPS 분류기를 유지하면서 정확도, latency, FPS를 함께 최적화

![](final_submission_assets_20260620/summary_key_results.svg)

## 1. 데이터셋 및 학습 조건

기본 데이터셋은 RPS 3-class 이미지 총 2717장이다. 클래스별 stratified split을 적용하여 train 2172장, validation 272장, test 273장으로 분리했다.

| class | total | train | val | test |
|---|---:|---:|---:|---:|
| scissors | 903 | 722 | 90 | 91 |
| rock | 907 | 725 | 91 | 91 |
| paper | 907 | 725 | 91 | 91 |
| total | 2717 | 2172 | 272 | 273 |

모델은 처음부터 random initialization으로 학습한 것이 아니라 `ImageNet` weight가 있는 `DenseNet121(weights='imagenet', include_top=False)`를 base model로 사용했다. 이후 RPS 데이터셋으로 fine-tuning하고, 최종 head는 3-class classifier로 학습했다.

## 2. 학습 Ablation 결과

초기 실험에서는 train/validation/test split을 명확히 분리한 뒤 augmentation 유무를 비교했다. baseline split만으로도 test 271/273을 달성했고, geometric/contrast 및 brightness augmentation 이후 test 272/273까지 올라갔다.

| 실험 | 핵심 설정 | test accuracy | TFLite dynamic quant accuracy | 모델 크기 |
|---|---|---:|---:|---:|
| 01 baseline split | augmentation 없음 | 271/273 = 99.27% | 99.27% | 7.08 MB |
| 02 geometric aug | rotation/translation/zoom/contrast | 272/273 = 99.63% | 99.63% | 7.08 MB |
| 03 brightness 10% | geometric + brightness 10% | 272/273 = 99.63% | 99.63% | 7.09 MB |

이후 최종 정확도 개선을 위해 current best top-2 모델을 대상으로 더 강한 augmentation을 적용했다.

사용한 최종 augmentation:

- `RandomFlip(horizontal)`
- `RandomRotation(0.12)`
- `RandomTranslation(0.12, 0.12)`
- `RandomZoom(0.15)`
- `RandomContrast(0.25)`
- `RandomBrightness(0.25)`
- `GaussianNoise(3.0)`

![](final_submission_assets_20260620/strong_aug_training_curves.svg)

최종 학습 결과, strong augmentation 기반 deploy Keras model이 validation 272/272, test 273/273을 달성했다. 이 모델에서 PTQ `random_c120_s31` 변환을 적용한 full-int8 TFLite가 최종 배포 후보가 되었다.

![](final_submission_assets_20260620/strong_aug_accuracy_latency.svg)

| 모델 | 변환 | test accuracy | Pi invoke mean | 판단 |
|---|---|---:|---:|---|
| 기존 best PTQ | full-int8 PTQ | 272/273 = 99.63% | 4.89 ms | 이전 best |
| 기존 best QAT | conv5/head QAT | 272/273 = 99.63% | 4.84 ms | 이전 best |
| strong PTQ random | full-int8 PTQ | 273/273 = 100.00% | 4.79 ms | 최종 선택 |
| strong PTQ brightness | full-int8 PTQ | 273/273 = 100.00% | 4.79 ms | 대체 후보 |
| strong QAT conv5/head | full-int8 QAT | 272/273 = 99.63% | 4.82 ms | 개선 없음 |

최종 모델:

```text
04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite
SHA256: C34199564DE4F7ADCFBA836D9A33BAE472739ACA8F1711C63E2E6335950FBBC7
```

## 3. Quantization 및 Raspberry Pi 실행 경로

Raspberry Pi 5의 CPU는 Cortex-A76 4-core이며 `asimd`, `asimddp`, `asimdhp` feature를 지원한다. 즉 하드웨어 수준에서는 NEON SIMD, INT8 dot-product, FP16 vector 연산 가능성이 있다. 하지만 실제 속도는 변환 형식만으로 결정되지 않고 TFLite delegate가 어떤 kernel 경로를 타는지가 더 중요했다.

초기 `tflite_runtime 2.14` 경로에서는 full-int8 모델에 XNNPACK delegate가 붙지 않아 오히려 dynamic quant보다 느렸다. 이후 TensorFlow 2.21 `tf.lite.Interpreter` 경로로 바꾸고 XNNPACK delegate를 사용하면서 full-int8 invoke가 5 ms 수준으로 내려갔다.

| 모델/경로 | input/output | delegate | test accuracy | best invoke |
|---|---|---:|---:|---:|
| float | float32/float32 | XNNPACK | 272/273 | 7.38 ms |
| dynamic quant | float32/float32 | XNNPACK hybrid | 272/273 | 6.94 ms |
| full-int8 초기 | int8/int8 | delegate 0 | 268/273 | 12.40 ms |
| full-int8 PTQ, TF backend | int8/int8 | XNNPACK | 268/273 | 5.02 ms |
| QAT sweep rep600 e30 | int8/int8 | XNNPACK | 272/273 | 6.04 ms |
| strong PTQ final | int8/int8 | XNNPACK | 273/273 | 4.79 ms |

중요한 해석은 다음과 같다. Dynamic quant가 빨라진 이유는 완전한 int8 activation pipeline 때문이라기보다 weight 크기 감소와 cache/memory bandwidth 이득이 컸다. 반대로 full-int8는 TensorFlow/XNNPACK 경로가 열렸을 때만 latency 이득이 뚜렷했다.

## 4. Raspberry Pi Real-Time Pipeline

실시간 앱은 Raspberry Pi에서 OpenCV camera frame을 읽고, 64x64 입력으로 전처리한 뒤 TFLite 모델을 호출하고 dashboard를 그리는 구조다.

대표 실행 조건:

- camera: USB UVC `/dev/video0`
- capture: 640x480 @ 30 FPS
- OpenCV buffer size: 4
- model input: 64x64x3 int8
- interpreter: TensorFlow 2.21 `tf.lite.Interpreter`
- delegate: XNNPACK CPU delegate
- threads: 4

신규 USB camera와 buffer size 4 적용 후 camera 공급은 안정적으로 30 FPS에 도달했다. 최종 full-int8 계열 pipeline은 전체 loop가 약 33.33 ms이며, FPS는 30.0 근처로 유지된다.

![](final_submission_assets_20260620/usb_camera_timing_pie_dashboard.svg)

| stage | mean time | 비율 |
|---|---:|---:|
| total loop | 33.33 ms | 100% |
| camera read/wait | 약 13.0 ms | 약 39% |
| preprocess | 약 1.6 ms | 약 5% |
| TFLite invoke | 약 5.0 ms | 약 15% |
| dashboard draw | 약 13.5 ms | 약 40% |
| FPS | 30.0 | - |

따라서 현재 E2E FPS 병목은 모델 단독 inference가 아니라 30 FPS camera cadence와 dashboard drawing이다. 모델 invoke는 5 ms 이하까지 줄었지만, camera가 30 FPS로 프레임을 공급하면 전체 loop는 약 33 ms보다 내려가기 어렵다.

## 5. DenseNet121 내부 Latency Breakdown

DenseNet121의 TFLite graph는 Conv2D 120개, Concatenation 58개, AveragePool 3개, FullyConnected 1개, Softmax 1개로 구성된다. Pi에 per-op profiler가 없어 prefix model을 여러 개 만들어 누적 latency를 측정하고 차분했다.

![](final_submission_assets_20260620/densenet_stage_breakdown_pie.svg)

| stage | latency share | clean 4.79 ms 기준 추정 |
|---|---:|---:|
| Stem + pool1 | 4.3% | 0.205 ms |
| Dense block 2 | 15.1% | 0.725 ms |
| Transition 2 | 3.5% | 0.169 ms |
| Dense block 3 | 23.4% | 1.120 ms |
| Transition 3 | 3.6% | 0.171 ms |
| Dense block 4 | 30.6% | 1.466 ms |
| Transition 4 | 4.5% | 0.214 ms |
| Dense block 5 | 13.2% | 0.633 ms |
| Final GAP/head | 1.9% | 0.091 ms |

핵심 병목은 Dense block 4와 Dense block 3이다. 두 block만 합쳐도 전체 invoke의 약 53.9%이며, dense block 전체는 약 82.3%를 차지한다. 따라서 final head나 softmax를 줄이는 것만으로는 큰 속도 개선이 어렵다.

![](final_submission_assets_20260620/densenet_stage_breakdown_bar.svg)

## 6. 추가 최적화 실험

QAT는 full-int8 정확도 회복에는 효과가 있었다. 단순 8 epoch QAT는 267/273으로 실패했지만, per-epoch TFLite validation 기준 선택과 representative set 600장 sweep을 통해 272/273까지 회복했다. 그러나 QAT graph에는 `QUANTIZE` op가 57개 남아 PTQ보다 약 1 ms 느렸다.

Pruning 50%, 75%도 시도했으나 최종 채택하지 않았다. 50% sparse-float는 test 272/273을 유지했지만 Pi invoke가 10.17 ms로 느렸고, sparse-int8는 accuracy가 크게 무너졌다. 원인은 TFLite graph에 `DENSIFY` op가 많이 생겨 sparse convolution fast path를 타지 못한 것으로 해석된다.

| 실험 | 결과 | 최종 판단 |
|---|---|---|
| QAT 8 epoch | 267/273, 6.09 ms | accuracy 부족 |
| QAT TFLite-best 20 epoch | 270/273, 6.09 ms | 개선되었지만 PTQ보다 느림 |
| QAT sweep rep600 e30 | 272/273, 6.04 ms | 정확도 회복, latency는 PTQ보다 느림 |
| pruning 50% sparse-float | 272/273, 10.17 ms | 느림 |
| pruning 75% sparse-float | 265/273, 10.11 ms | 정확도/속도 모두 부족 |
| strong augmentation PTQ | 273/273, 4.79 ms | 최종 선택 |

## 7. 카메라 실험

초기 webcam pipeline은 camera read 자체가 약 15 FPS 수준으로 묶였고, 모델보다 camera I/O가 더 큰 병목이었다. 이후 새로운 USB UVC camera를 연결하고 OpenCV buffer size를 4로 설정하면서 640x480@30 FPS를 안정적으로 얻었다.

CSI IMX219 camera도 테스트했다. `rpicam-hello --list-cameras`에서는 IMX219 sensor와 640x480 @ 103 FPS mode가 감지되었다. 하지만 실제 streaming 단계에서 `Camera frontend has timed out`, `Error writing reg 0x0100`가 반복되어 첫 프레임을 받지 못했다. 두 connector를 모두 확인했지만 동일하게 실패했으므로, 현재 제출 실측은 USB UVC camera 기준으로 정리한다.

CSI camera가 정상화된다면 640x480 @ 103 FPS mode가 가능 후보이며, 이 경우 현재 4.79 ms inference는 60 FPS 이상 pipeline에서도 충분한 headroom을 가진다. 다만 dashboard drawing과 frame acquisition 방식을 함께 최적화해야 한다.

## 8. 최종 결론

최종 제출 모델은 `04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite`이다. DenseNet121 구조를 유지하면서 strong augmentation과 PTQ calibration을 적용했고, held-out test set에서 273/273, 100.00%를 달성했다. Raspberry Pi 5에서는 TensorFlow/XNNPACK full-int8 경로로 평균 invoke 4.79 ms를 기록했다.

실시간 pipeline은 USB UVC camera 기준 30 FPS로 동작한다. 현재 전체 FPS의 직접 병목은 모델이 아니라 camera cadence와 dashboard drawing이다. DenseNet 내부에서는 Dense block 3/4가 가장 무거우며, DenseNet을 유지하는 조건에서는 runtime backend, thread, UI drawing, camera path 최적화가 현실적인 개선 방향이다.

최종 백업 위치:

```text
[LOCAL_WORKSPACE]\best_models_backup_20260618_1833_strong_aug
[OBSIDIAN_COURSEWORK]\2026_0618\best_models_backup_20260618_1833_strong_aug
```

