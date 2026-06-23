# QAT sweep more runs

- 실험 시각: 2026-06-18 10:24-13:02 KST
- GPU 서버: `codex-isl-gpu1`, GPU 0 한 장 사용
- 목적: QAT를 더 많이 돌리고, learning rate와 representative set을 바꿔 full-int8 TFLite 정확도를 보완
- 최종 best 모델: `03_aug_geometric_brightness10.qat_sweep_rep600_e30_full_int8_io.tflite`

## 핵심 결론

이번 sweep에서 full-int8 TFLite test accuracy가 float 수준까지 회복됐다.

| 모델 | test accuracy | 정답/전체 | dashboard invoke |
|---|---:|---:|---:|
| PTQ full-int8 | 98.17% | 268/273 | 5.0216 ms |
| QAT 8epoch val-loss | 97.80% | 267/273 | 6.0906 ms |
| QAT 20epoch TFLite-best | 98.90% | 270/273 | 6.0875 ms |
| QAT sweep rep600 e30 | 99.63% | 272/273 | 6.0397 ms |
| Float/Dynamic 계열 | 99.63% | 272/273 | 별도 |

![](qat_sweep_20260618_1024/figures/accuracy_compare.svg)

## Sweep 설정과 결과

| variant | lr | epoch | rep | best val | best epoch | TFLite test | 비고 |
|---|---:|---:|---:|---:|---:|---:|---|
| `lr5e-6_e30_rep300_pat6` | 5e-06 | 30 | 300 | 99.63% (271/272) | 26 | 98.53% (269/273) |  |
| `lr2e-5_e24_rep300_pat4` | 2e-05 | 24 | 300 | 100.00% (272/272) | 20 | 99.27% (271/273) |  |
| `lr1e-5_e30_rep600_pat6` | 1e-05 | 30 | 600 | 100.00% (272/272) | 30 | 99.63% (272/273) | best |

![](qat_sweep_20260618_1024/figures/sweep_tflite_val_accuracy.svg)

가장 중요한 관찰은 representative 600장 + 30 epoch 조합이 test set에서 가장 안정적으로 맞았다는 점이다. 이전 QAT는 validation에서는 좋아도 test에서 270/273에 머물렀는데, 이번 rep600 모델은 test에서도 272/273까지 올라갔다.

## Best 모델 상세

| 항목 | 값 |
|---|---:|
| best variant | `lr1e-5_e30_rep600_pat6` |
| best epoch | 30 |
| TFLite val accuracy | 100.00% (272/272) |
| QAT Keras test accuracy | 100.00% |
| full-int8 TFLite test accuracy | 99.63% (272/273) |
| TFLite size | 7.097 MB |

Confusion matrix:

```text
true\pred   scissors  rock  paper
scissors         90     1     0
rock              0    91     0
paper             0     0    91
```

오분류는 scissors 한 장이 rock으로 간 것뿐이다. 이 결과는 float model의 test confusion과 같은 수준이다.

## Pi latency

Pure invoke:

| threads | invoke mean | total mean | delegate | graph |
|---:|---:|---:|---:|---|
| 1 | 5.9969 ms | 6.0201 ms | 4 | CONV_2D 120, QUANTIZE 57 |
| 2 | 6.0200 ms | 6.0451 ms | 4 | CONV_2D 120, QUANTIZE 57 |
| 4 | 5.8644 ms | 5.8904 ms | 4 | CONV_2D 120, QUANTIZE 57 |

Camera pipeline, dashboard draw:

![](qat_sweep_20260618_1024/figures/best_timing_dashboard_pie.svg)

| 항목 | mean | loop 비율 |
|---|---:|---:|
| total loop | 33.3286 ms | 100.00% |
| FPS | 30.004 | - |
| camera read | 11.6278 ms | 34.89% |
| dashboard draw | 13.8414 ms | 41.53% |
| tflite invoke | 6.0397 ms | 18.12% |
| predict total | 6.1584 ms | 18.48% |

No-dashboard invoke는 6.8448 ms이고, dashboard 조건 invoke는 6.0397 ms이다. QAT graph에는 여전히 `QUANTIZE` op 57개가 남아 있어서 PTQ full-int8보다 약 1 ms 느리다. 즉, 이번 sweep은 latency 개선보다는 accuracy 복구 실험으로 성공했다.

## 결론

- accuracy 기준 best는 `lr1e-5_e30_rep600_pat6`이다.
- full-int8 TFLite 정확도는 272/273, 99.63%로 float/dynamic과 동률까지 올라왔다.
- latency는 PTQ full-int8가 여전히 빠르지만, QAT sweep 모델은 정확도 손실 없이 full-int8 I/O를 쓸 수 있는 후보가 됐다.
- 실시간 카메라 앱의 전체 FPS는 여전히 30 fps 근처이고, 병목은 camera cadence와 dashboard draw 쪽이다.

## 실행/백업

- Pi 실행: `./07_RUN_QAT_SWEEP_REP600_E30_WINDOW.sh`
- Pi 모델: `generated_models/codex_ablation_20260617_1558/03_aug_geometric_brightness10.qat_sweep_rep600_e30_full_int8_io.tflite`
- 로컬 artifact: `[LOCAL_WORKSPACE]\qat_sweep_20260618_1024`
- GPU artifact: `~[TRAINING_WORKSPACE]/rps_ablation_20260617_1558/results/qat_sweep_20260618_1024`

