# CSI ov5647 + Best Model Dashboard / Timing Breakdown

- 측정 시각: 2026-06-22 09:38 KST
- Device: Raspberry Pi 5
- Camera: CSI `ov5647`
- 입력 해상도: `640x480`
- 요청 FPS: `58`
- 모델: `04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite`
- 모델 SHA256: `c34199564de4f7adcfba836d9a33bae472739aca8f1711c63e2e6335950fbbc7`

## TL;DR

현재 best 모델은 strong augmentation 후 PTQ를 적용한 full-int8 모델이다.

```text
generated_models/codex_ablation_20260617_1558/strong_aug_top2_20260618_1752/04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite
```

새 CSI 카메라 `ov5647`은 정상 인식 및 스트리밍에 성공했다. `640x480@58` 설정에서 최종 모델 포함 no-dashboard loop는 평균 `15.05 ms`, 약 `66.5 FPS`로 측정됐다. Dashboard draw까지 포함하면 평균 `23.74 ms`, 약 `42.1 FPS`이다.

Pi 화면 dashboard는 아래 설정으로 실행했다.

```bash
python -u scripts/pi_rps_window_dashboard.py \
  --model generated_models/codex_ablation_20260617_1558/strong_aug_top2_20260618_1752/04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite \
  --camera 0 \
  --camera-backend rpicam \
  --capture-width 640 \
  --capture-height 480 \
  --capture-fps 58 \
  --interpreter-backend tensorflow \
  --num-threads 4 \
  --window-scale 1.0 \
  --threshold 0.65 \
  --stable-frames 8 \
  --mirror
```

실행 확인:

- smoke test: 성공
- live dashboard launch: 성공
- launcher PID: `1820`
- dashboard worker PID: `1838`
- rpicam process PID: `1849`
- log: `timing_results/csi_ov5647_20260622/dashboard_live_detached_20260622_094339.log`
- 실행 방식: `setsid`로 SSH 세션과 분리하여 실행했으며, 새 SSH 세션에서 프로세스 생존을 재확인했다.

## Camera 확인

`rpicam-hello --list-cameras` 결과에서 `ov5647`이 정상 인식됐다.

| Mode | Max FPS |
|---|---:|
| 640x480 | 58.92 |
| 1296x972 | 46.34 |
| 1920x1080 | 32.81 |
| 2592x1944 | 15.63 |

`rpicam-jpeg`로 샘플 이미지 캡처도 성공했다.

![](csi_ov5647_best_timing_20260622_0938/ov5647_640x480_sample.jpg)

샘플은 어둡게 나왔으므로 실제 demo에서는 조명/노출/렌즈 방향을 확인하는 것이 좋다.

## Camera-Only FPS Probe

`rpicam-vid` MJPEG path에서 4초 측정했다.

| Case | frames | measured FPS | read mean |
|---|---:|---:|---:|
| 640x480 @ 58 | 266 | 66.41 | 15.04 ms |
| 1296x972 @ 46 | 199 | 49.69 | 20.10 ms |
| 1920x1080 @ 32 | 139 | 34.61 | 28.87 ms |
| 2592x1944 @ 15 | 66 | 16.34 | 61.18 ms |

`640x480`은 60 FPS급 pipeline에 사용할 수 있는 상태다.

## Timing Breakdown: No Dashboard

조건:

- camera backend: `rpicam:0:mjpeg`
- capture: `640x480@58`
- interpreter: TensorFlow `tf.lite.Interpreter`
- delegate: XNNPACK CPU delegate
- threads: 4
- runs: 500
- warmup: 50

![](csi_ov5647_best_timing_20260622_0938/timing_pie_nodashboard.svg)

| Stage | mean | p95 | share |
|---|---:|---:|---:|
| total loop | 15.0455 ms | 15.6605 ms | 100.00% |
| camera read | 8.3832 ms | 8.9238 ms | 55.72% |
| preprocess total | 1.5070 ms | 1.6601 ms | 10.02% |
| TFLite invoke | 5.0031 ms | 5.1966 ms | 33.25% |
| predict total | 5.1202 ms | 5.3089 ms | 34.03% |

Derived:

```text
E2E FPS from total loop mean: 66.465 FPS
```

해석: dashboard가 없으면 카메라 read/wait와 model invoke가 거의 loop를 나눠 가진다. 기존 USB 30 FPS camera 대비 frame budget이 절반 이하로 줄었고, 이제 inference가 의미 있는 병목으로 보인다.

## Timing Breakdown: Dashboard Draw

이 측정은 실제 `imshow` 출력은 제외하고 dashboard image composition 비용만 포함했다. 실제 화면 출력은 compositor/display 상태에 따라 추가 변동이 생길 수 있다.

![](csi_ov5647_best_timing_20260622_0938/timing_pie_dashboard_draw.svg)

| Stage | mean | p95 | share |
|---|---:|---:|---:|
| total loop | 23.7449 ms | 24.0827 ms | 100.00% |
| camera read | 2.8143 ms | 2.9053 ms | 11.85% |
| preprocess total | 1.5889 ms | 1.7052 ms | 6.69% |
| TFLite invoke | 5.1377 ms | 5.3776 ms | 21.64% |
| predict total | 5.2592 ms | 5.5019 ms | 22.15% |
| dashboard draw, no imshow | 14.0419 ms | 14.2072 ms | 59.14% |

Derived:

```text
E2E FPS from total loop mean: 42.114 FPS
```

해석: dashboard를 그리는 순간 병목은 모델이 아니라 UI drawing으로 바뀐다. Dashboard draw가 약 `14.0 ms`로 전체의 약 `59%`를 차지한다. 실제 시연에서 dashboard가 필요하면 40 FPS급, headless/inference-only면 60 FPS급으로 보는 것이 맞다.

## 이전 USB Camera 대비

![](csi_ov5647_best_timing_20260622_0938/pipeline_speed_compare.svg)

| 조건 | total loop | FPS | 핵심 병목 |
|---|---:|---:|---|
| USB UVC dashboard | 33.33 ms | 30.0 FPS | camera cadence + dashboard draw |
| CSI ov5647 no dashboard | 15.05 ms | 66.5 FPS | camera read + inference |
| CSI ov5647 dashboard draw | 23.74 ms | 42.1 FPS | dashboard draw |

이번 CSI 카메라 교체로 기존 30 FPS camera 병목은 실질적으로 해소됐다. 다만 dashboard를 켜면 UI drawing이 새 병목이 된다.

## 결론

최종 best 모델은 그대로 유지한다.

```text
04_aug_strong_flip_light25.ptq_random_c120_s31_full_int8_io.tflite
```

새 CSI `ov5647` 카메라를 사용하면 `640x480` 기준 모델 포함 pipeline이 no-dashboard에서 약 `66 FPS`까지 올라간다. 즉 camera I/O는 더 이상 30 FPS로 막히지 않는다. 현재 속도 최적화 관점의 우선순위는 다음과 같다.

1. 시연용 dashboard가 필요하면 drawing cost를 줄인다.
2. 실제 FPS만 중요하면 dashboard를 끄고 headless/inference path로 실행한다.
3. 더 높은 FPS를 원하면 DenseNet invoke 약 `5.0 ms`를 줄이는 방향이 필요하다.

산출물 위치:

```text
Pi: timing_results/csi_ov5647_20260622/
Local/Obsidian asset: csi_ov5647_best_timing_20260622_0938/
```

