# RPS pruning 50/75 실험 보고서

- 실험 시작: 2026-06-18 17:10 KST
- 목적: 현재 best 모델을 백업한 뒤, pruning sparsity 50%와 75%가 Raspberry Pi 실제 TFLite invoke latency를 줄이는지 확인
- 기준 모델: `03_aug_geometric_brightness10.best.keras`
- 평가 test set: 273장
- Pi 측정 환경: Raspberry Pi, TensorFlow 2.21.0, TFLite XNNPACK CPU delegate, thread 1/2/4, 200 runs, 30 warmup

## 결론

이번 pruning 50%/75% 실험은 최종 채택하지 않는 것이 좋다.

현재 best는 그대로 유지한다. 50% sparse-float는 test accuracy 99.63%를 유지했지만 Pi 4-thread invoke가 10.17 ms로 best QAT 4.77 ms보다 약 2.1배 느리다. 75% sparse-float는 파일 크기는 줄었지만 accuracy가 97.07%로 떨어지고 invoke도 10.11 ms로 느리다. sparse-int8 모델들은 invoke 시간은 best와 비슷하지만 accuracy가 46.89%, 67.40%로 무너졌다.

가장 중요한 원인은 TFLite op에 `DENSIFY`가 121개 들어간 점이다. 즉 sparse weight가 실제 sparse convolution fast path로 바로 처리되는 것이 아니라, 런타임에서 dense 형태로 풀리는 경로가 생겼다. 그래서 파일 크기 압축 효과는 있어도 Raspberry Pi latency 개선으로 이어지지 않았다.

## 현재 best 백업

백업 위치:

- Local: `[LOCAL_WORKSPACE]\best_models_backup_20260618_1710`
- Obsidian: `[OBSIDIAN_COURSEWORK]\2026_0618\best_models_backup_20260618_1710`

| 모델 | 역할 | test accuracy | Pi 4-thread invoke mean | SHA256 |
|---|---:|---:|---:|---|
| `03_aug_geometric_brightness10.ptq_random_c120_s31_full_int8_io.tflite` | best PTQ | 272/273 = 99.63% | 4.88 ms | `6F901C8504E06C8258C0E93384C68450C8EB3E132BC4842F3333597E5A5ADD80` |
| `03_aug_geometric_brightness10.qat_conv5_head_e18_full_int8_io.tflite` | best QAT | 272/273 = 99.63% | 4.77 ms | `E106A5C876B7C9C2D8B9CCF6A36DC6A2D5E320C6361E1E66C26D989D9533F5B8` |

## 실험 방법

TensorFlow Model Optimization Toolkit pruning을 사용했다. 먼저 XNNPACK latency용 pruning policy인 `PruneForLatencyOnXNNPack()`을 시도했지만, 현재 DenseNet 기반 모델 구조가 policy 조건을 통과하지 못해 generic `prune_low_magnitude` fallback으로 수행됐다.

모델 구조 호환을 위해 head를 다음처럼 재구성했다.

- DenseNet feature extractor weight transfer
- `GlobalAveragePooling2D(keepdims=True)`
- `Flatten`
- `Dropout`
- Dense 3-class head

각 sparsity에 대해 8 epoch fine-tuning을 수행했다.

- sparsity: 50%, 75%
- learning rate: `1e-5`
- batch size: 32
- representative dataset: 120장, seed 31
- 변환 1: sparse float, `Optimize.EXPERIMENTAL_SPARSITY`
- 변환 2: sparse full-int8 IO, `Optimize.DEFAULT + Optimize.EXPERIMENTAL_SPARSITY`

## 학습 곡선

![](pruning_xnnpack_20260618_1710/figures/pruning_training_curves.svg)

50% pruning은 validation accuracy가 안정적으로 98-99%대를 유지했다. 75% pruning은 validation loss와 accuracy가 크게 흔들렸고, 최종 test accuracy도 97.07%로 낮아졌다. 즉 75%는 압축률은 높지만 모델 안정성이 떨어진다.

## 정확도 및 Pi latency

![](pruning_xnnpack_20260618_1710/figures/pruning_latency_accuracy_tradeoff.svg)

| 모델 | dtype | total sparsity | test accuracy | size | Pi 4-thread invoke mean | delegate | 주요 op 특징 |
|---|---:|---:|---:|---:|---:|---:|---|
| best QAT conv5/head | int8 | 0% | 272/273 = 99.63% | 7.06 MB | 4.77 ms | 4 | `QUANTIZE` 15개 |
| best PTQ random c120 s31 | int8 | 0% | 272/273 = 99.63% | 7.04 MB | 4.88 ms | 4 | no extra quantize |
| prune-compatible PTQ baseline | int8 | 0% | 272/273 = 99.63% | 7.54 MB | 4.80 ms | 4 | `DENSIFY` 5개 |
| pruned 50 sparse-float | float32 | 48.80% | 272/273 = 99.63% | 19.05 MB | 10.17 ms | 1 | `DENSIFY` 121개 |
| pruned 75 sparse-float | float32 | 73.19% | 265/273 = 97.07% | 9.81 MB | 10.11 ms | 1 | `DENSIFY` 121개 |
| pruned 50 sparse-int8 | int8 | 48.80% | 128/273 = 46.89% | 9.40 MB | 4.82 ms | 4 | accuracy collapse |
| pruned 75 sparse-int8 | int8 | 73.19% | 184/273 = 67.40% | 5.07 MB | 4.92 ms | 4 | accuracy collapse |

## 관찰

1. sparse-float는 정확도 보존에는 유리하지만 속도가 너무 느리다.
2. sparse-int8는 속도가 best int8와 거의 같지만 정확도가 크게 무너졌다.
3. 75% pruning은 float에서도 정확도가 97.07%로 하락한다.
4. `DENSIFY` op가 많이 생긴 것으로 보아, Pi에서 sparse convolution acceleration을 제대로 타지 못한다.
5. pruning은 여기서는 latency 최적화보다 파일 크기 압축 실험에 가깝게 동작했다.

## 파일 및 해시

| 파일 | SHA256 |
|---|---|
| `baseline_prune_compatible_ptq_c120_s31_full_int8_io.tflite` | `2B1BA6BCBFFFD8E150349319C5676BED11A2269ACD3007C74D7602A12ABF5FB9` |
| `03_aug_geometric_brightness10.pruned_sparsity_50_sparse_float.tflite` | `67CD3335B823A99B86519860A3F1F85126D876DC28C83F35877230C80995CB69` |
| `03_aug_geometric_brightness10.pruned_sparsity_50_sparse_full_int8_io.tflite` | `4B53B05B02AFCA336197EB428BCE548DF065F8A03A74467F09755BF86388FF10` |
| `03_aug_geometric_brightness10.pruned_sparsity_75_sparse_float.tflite` | `45ECFD5BFB2E77C24B79276EA2F315DF58427629D37A3893C8BF1399A5EB2273` |
| `03_aug_geometric_brightness10.pruned_sparsity_75_sparse_full_int8_io.tflite` | `D69C0361205943183D6CA79A17212CF984D1815FF32609776B96FA9BB9B9DD21` |

## 최종 판단

현재 제출/데모용 best는 `best_qat_conv5` 또는 `best_ptq`를 유지한다. 둘 다 99.63% 정확도이며 Pi invoke는 약 4.8 ms이다.

pruning으로 더 빠르게 만들려면 다음 방향이 필요하다.

1. 현재 DenseNet을 계속 pruning하기보다, 처음부터 Pi/XNNPACK에 잘 맞는 작은 모델로 architecture search를 수행한다.
2. pruning을 계속 연구하려면 XNNPACK sparse fast path 조건을 만족하는 구조로 모델을 다시 설계하고, `DENSIFY`가 대량으로 생기지 않는지 먼저 확인한다.
3. pruned-int8 정확도 회복은 pruning 후 PTQ보다 pruning-aware QAT가 필요하지만, 이번 결과 기준으로는 latency 이득이 없어서 우선순위는 낮다.


