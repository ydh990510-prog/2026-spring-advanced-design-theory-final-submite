# RPS ResNet50 TFLite 생성 노트북 실행 과정

## 실행 순서

1. `00_train_baseline_resnet50.ipynb`
   - 공통 baseline float 모델 생성
   - 산출물: `baseline_resnet50_float.h5`

2. `01_resnet50_qat_only.ipynb`
   - baseline load
   - QAT only
   - 산출물: `RPS_ResNet50_QAT_only.tflite`

3. `02_resnet50_clustering_qat.ipynb`
   - baseline load
   - clustering → cluster-preserving QAT
   - 산출물: `RPS_ResNet50_Clustering_QAT.tflite`

4. `03_resnet50_pruning_qat.ipynb`
   - baseline load
   - pruning → sparsity-preserving QAT
   - 산출물: `RPS_ResNet50_Pruning_QAT.tflite`

5. `04_resnet50_pruning_clustering_qat.ipynb`
   - baseline load
   - pruning → sparsity-preserving clustering → QAT
   - 산출물: `RPS_ResNet50_Pruning_Clustering_QAT.tflite`

## 학습 과정 중, OOM 방지법

- 각 노트북을 따로 실행합니다.
- 한 노트북이 끝나면 커널을 재시작한 뒤 다음 노트북을 실행힙니다.
- OOM 발생 시, 각 노트북 상단의 `BATCH_SIZE`를 4 → 2로 낮추고 다시 실행합니다.
- 모든 실험은 같은 `baseline_resnet50_float.h5`에서 출발합니다.

## 경로 설정

다른 경로라면 노트북 상단의 `DEFAULT_DATASET_DIR`를 수정하거나,
터미널에서 아래처럼 지정하여 실행합니다.
[BASH]
export RPS_DATASET_DIR=/your/path/RPS_Dataset
export RPS_SAVE_DIR=/your/save/path

## 학습 및 개발용 모델 파일(.HDF)을 Google drive에 업로드하였습니다.

[Google drive 주소]
https://drive.google.com/file/d/1BBzbJ5MzWsYoi16JOhrXErmBPJa2I3mY/view?usp=sharing
