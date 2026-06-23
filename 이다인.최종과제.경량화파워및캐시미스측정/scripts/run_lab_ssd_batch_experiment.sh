#!/usr/bin/env bash
set -euo pipefail

MODE="ssd_detection"
DURATION="60"
WARMUP="10"
REPEATS="3"
COOLDOWN="120"
PREPARE_SECONDS="5"
CONFIDENCE_THRESHOLD="0.5"
OUTPUT_ROOT="experiments/batch_runs"
EXPERIMENT_NAME="professor_lab_ssd_measurement"
PYTHON="${HOME}/camera_test/.venv311/bin/python"
NO_PROMPT="0"
RESUME_BATCH="0"
START_INDEX="1"
BATCH_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/run_lab_ssd_batch_experiment.sh [options]

Options:
  --mode MODE                    ssd_detection or ssd_no_display (default ssd_detection)
  --duration SECONDS             measured duration per run (default 60)
  --warmup SECONDS               warmup duration per run (default 10)
  --repeats N                    repeats per condition (default 3)
  --cooldown SECONDS             cooldown between runs (default 120)
  --prepare-seconds SECONDS      countdown before gesture runs (default 5)
  --confidence-threshold VALUE   detection threshold (default 0.5)
  --output-root DIR              root for batch folders (default experiments/batch_runs)
  --experiment-name NAME         batch name prefix (default professor_lab_ssd_measurement)
  --python PATH                  Python executable (default ~/camera_test/.venv311/bin/python)
  --no-prompt                    do not prompt before gesture runs
  --resume-batch                 resume existing latest or specified batch folder
  --batch-dir DIR                batch folder to resume/use
  --start-index N                skip manifest rows before this 1-based index
  -h, --help                     show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --prepare-seconds) PREPARE_SECONDS="$2"; shift 2 ;;
    --confidence-threshold) CONFIDENCE_THRESHOLD="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --no-prompt) NO_PROMPT="1"; shift ;;
    --resume-batch) RESUME_BATCH="1"; shift ;;
    --batch-dir) BATCH_DIR="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$MODE" != "ssd_detection" && "$MODE" != "ssd_no_display" ]]; then
  echo "--mode must be ssd_detection or ssd_no_display" >&2
  exit 2
fi

if [[ ! -f "scripts/run_lab_ssd_measurement.py" ]]; then
  echo "Run this script from the project root." >&2
  exit 2
fi

if [[ -z "$BATCH_DIR" ]]; then
  if [[ "$RESUME_BATCH" == "1" ]]; then
    BATCH_DIR="$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "${EXPERIMENT_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "$BATCH_DIR" ]]; then
      echo "No existing batch folder found under $OUTPUT_ROOT for $EXPERIMENT_NAME" >&2
      exit 2
    fi
  else
    BATCH_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
  fi
fi

RAW_DIR="${BATCH_DIR}/raw_runs"
MANIFEST="${BATCH_DIR}/manifest.csv"
mkdir -p "$RAW_DIR"

if [[ ! -f "$MANIFEST" ]]; then
  echo "run_index,condition_mode,expected_label,repeat,status,run_dir,started_at,ended_at" > "$MANIFEST"
fi

run_exists() {
  local condition_mode="$1"
  local expected_label="$2"
  local repeat="$3"
  find "$RAW_DIR" -mindepth 2 -maxdepth 3 -type f \
    -path "${RAW_DIR}/${condition_mode}_${expected_label}_r${repeat}_*/**/summary.json" \
    -print -quit | grep -q .
}

append_manifest() {
  local run_index="$1"
  local condition_mode="$2"
  local expected_label="$3"
  local repeat="$4"
  local status="$5"
  local run_dir="$6"
  local started_at="$7"
  local ended_at="$8"
  echo "${run_index},${condition_mode},${expected_label},${repeat},${status},${run_dir},${started_at},${ended_at}" >> "$MANIFEST"
}

conditions=()
for ((r=1; r<=REPEATS; r++)); do conditions+=("idle none $r"); done
for ((r=1; r<=REPEATS; r++)); do conditions+=("camera_only none $r"); done
for label in scissors rock paper; do
  for ((r=1; r<=REPEATS; r++)); do conditions+=("$MODE $label $r"); done
done

total="${#conditions[@]}"
run_index=0
for condition in "${conditions[@]}"; do
  run_index=$((run_index + 1))
  if (( run_index < START_INDEX )); then
    continue
  fi

  read -r condition_mode expected_label repeat <<< "$condition"
  if [[ "$RESUME_BATCH" == "1" ]] && run_exists "$condition_mode" "$expected_label" "$repeat"; then
    echo "[$run_index/$total] Skipping existing $condition_mode $expected_label repeat $repeat"
    append_manifest "$run_index" "$condition_mode" "$expected_label" "$repeat" "skipped_existing" "" "" ""
    continue
  fi

  echo "[$run_index/$total] Next: mode=$condition_mode expected=$expected_label repeat=$repeat"
  if [[ "$expected_label" != "none" && "$NO_PROMPT" != "1" ]]; then
    read -r -p "Prepare gesture '${expected_label}', then press Enter to start. "
  fi

  if [[ "$expected_label" != "none" && "$PREPARE_SECONDS" != "0" ]]; then
    echo "Starting in $PREPARE_SECONDS seconds..."
    sleep "$PREPARE_SECONDS"
  fi

  run_output="${RAW_DIR}/${condition_mode}_${expected_label}_r${repeat}_$(date +%Y%m%d_%H%M%S)"
  started_at="$(date -Iseconds)"
  "$PYTHON" scripts/run_lab_ssd_measurement.py \
    --mode "$condition_mode" \
    --duration "$DURATION" \
    --warmup "$WARMUP" \
    --expected-label "$expected_label" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --output-dir "$run_output"
  ended_at="$(date -Iseconds)"
  summary_path="$(find "$run_output" -type f -name summary.json -print -quit)"
  if [[ -n "$summary_path" ]]; then
    actual_run_dir="$(dirname "$summary_path")"
  else
    actual_run_dir="$run_output"
  fi
  append_manifest "$run_index" "$condition_mode" "$expected_label" "$repeat" "completed" "$actual_run_dir" "$started_at" "$ended_at"

  if (( run_index < total )) && [[ "$COOLDOWN" != "0" ]]; then
    echo "Cooling down for $COOLDOWN seconds..."
    sleep "$COOLDOWN"
  fi
done

"$PYTHON" scripts/aggregate_lab_ssd_measurements.py "$RAW_DIR"
echo "Batch complete: $BATCH_DIR"
