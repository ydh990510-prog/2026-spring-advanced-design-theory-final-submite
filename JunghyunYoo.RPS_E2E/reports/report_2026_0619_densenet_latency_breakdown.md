# DenseNet121 세부 latency breakdown (2026-06-19)

## 결론

현재 best 모델 `04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite` 기준으로 DenseNet 내부를 prefix 모델로 잘라 측정했다. Pi에 `benchmark_model` op profiler가 없어서, 각 DenseNet endpoint까지 변환한 TFLite 모델의 누적 latency를 재고 차분을 냈다.

핵심 병목은 `Dense block 4`와 `Dense block 3`이다. 같은 run 기준 `Dense block 4`가 30.6%, `Dense block 3`가 23.4%이고, 둘을 합치면 약 53.9%다. 전체 dense block을 모두 합치면 82.3%라서, softmax/head를 만지는 것보다 DenseNet convolution 구간 자체가 속도를 결정한다.

## 측정 조건

- Device: Raspberry Pi, TensorFlow `2.21.0`
- Model: full int8 I/O TFLite, XNNPACK delegate 사용
- Threads: 4
- Runs: warmup 50, measured 400
- Prefix endpoints: stem, dense2/3/4/5 내부 chunk, transition, final head
- 같은 긴 prefix run에서 실제 full model invoke mean: `6.503 ms`
- 별도 재측정 full model invoke mean: `5.302 ms`
- 이전 clean-ish best 기록: `4.793 ms`

주의: prefix 전체를 연속으로 돌린 run에서는 Pi 온도가 올라서 실제 full model이 `6.503 ms`까지 느려졌다. 그래서 아래 raw ms는 “같은 run 내부 비율” 판단에 가장 적합하고, 실사용 절대값은 clean/reference 기준으로 스케일해서 보는 게 더 안전하다.

## Stage Breakdown

![DenseNet stage pie](densenet_prefix_profile_20260619/figures/densenet_stage_breakdown_pie.svg)

![DenseNet stage bar](densenet_prefix_profile_20260619/figures/densenet_stage_breakdown_bar.svg)

| Stage | Raw diff ms | Share | Clean 4.79ms est | Ops |
| --- | --- | --- | --- | --- |
| Stem + pool1 | 0.277 | 4.3% | 0.205 | 1 Conv, 1 MaxPool |
| Dense block 2 | 0.984 | 15.1% | 0.725 | 12 Conv, 6 Concat |
| Transition 2 | 0.229 | 3.5% | 0.169 | 1 Conv, 1 AvgPool |
| Dense block 3 | 1.519 | 23.4% | 1.120 | 24 Conv, 12 Concat |
| Transition 3 | 0.232 | 3.6% | 0.171 | 1 Conv, 1 AvgPool |
| Dense block 4 | 1.989 | 30.6% | 1.466 | 48 Conv, 24 Concat |
| Transition 4 | 0.290 | 4.5% | 0.214 | 1 Conv, 1 AvgPool |
| Dense block 5 | 0.859 | 13.2% | 0.633 | 32 Conv, 16 Concat |
| Final ReLU/GAP/head | 0.123 | 1.9% | 0.091 | Mean, FC, Softmax |

## 더 세분화한 Chunk Breakdown

Dense block 내부를 몇 개 layer 단위로 더 쪼갠 결과다. 특히 Dense4의 뒤쪽 chunk만 특별히 압도적인 것이 아니라, Dense4 전체가 꾸준히 무겁고 Dense3/Dense5도 의미 있게 누적된다.

| Chunk | Raw diff ms | Share |
| --- | --- | --- |
| Stem + pool1 | 0.277 | 4.3% |
| Dense2 blocks 01-03 | 0.434 | 6.7% |
| Dense2 blocks 04-06 | 0.549 | 8.4% |
| Transition 2 | 0.229 | 3.5% |
| Dense3 blocks 01-04 | 0.484 | 7.4% |
| Dense3 blocks 05-08 | 0.463 | 7.1% |
| Dense3 blocks 09-12 | 0.573 | 8.8% |
| Transition 3 | 0.232 | 3.6% |
| Dense4 blocks 01-06 | 0.439 | 6.7% |
| Dense4 blocks 07-12 | 0.467 | 7.2% |
| Dense4 blocks 13-18 | 0.566 | 8.7% |
| Dense4 blocks 19-24 | 0.517 | 8.0% |
| Transition 4 | 0.290 | 4.5% |
| Dense5 blocks 01-04 | 0.099 | 1.5% |
| Dense5 blocks 05-08 | 0.196 | 3.0% |
| Dense5 blocks 09-12 | 0.309 | 4.7% |
| Dense5 blocks 13-16 | 0.256 | 3.9% |
| Final ReLU/GAP/head | 0.123 | 1.9% |

## Prefix Cumulative Curve

![DenseNet prefix cumulative](densenet_prefix_profile_20260619/figures/densenet_prefix_cumulative.svg)

`17_final_relu`, `18_global_average_pool`, `19_softmax_head` endpoint는 일부 non-monotonic noise가 있었다. 별도 TFLite 모델로 endpoint output을 바꾸면 delegate partition과 output copy 조건이 달라질 수 있어서 tail은 `actual full same run - dense5_block16 endpoint`로 계산했다.

## 해석

- DenseNet121은 총 `120 Conv2D`, `58 Concatenation`, `3 AveragePool`, `1 Mean`, `1 FullyConnected`, `1 Softmax`로 구성되어 있다.
- Dense block 구간만 약 82.3%다. 즉 latency 대부분은 Conv2D와 concat이 있는 dense connectivity에서 나온다.
- Transition block 세 개는 합쳐도 11.6% 수준이다.
- Stem은 4.3%이고, final ReLU/GAP/head는 약 1.9%로 작다.
- 따라서 “아주 빠르게” 만들려면 head 최적화보다 Dense3/Dense4 계산량을 줄이거나, 입력 해상도/연산 backend/스레드와 thermal 조건을 조정하는 쪽이 효과가 크다.

## 산출물

- Raw prefix benchmark: `densenet_prefix_profile_20260619/prefix_thread_probe_4threads.pi.json`
- Full model recheck: `densenet_prefix_profile_20260619/actual_strong_ptq_recheck.pi.json`
- Stage CSV: `densenet_prefix_profile_20260619/densenet_stage_breakdown_summary.csv`
- Fine chunk CSV: `densenet_prefix_profile_20260619/densenet_fine_chunk_breakdown.csv`
- Prefix cumulative CSV: `densenet_prefix_profile_20260619/densenet_prefix_cumulative.csv`

