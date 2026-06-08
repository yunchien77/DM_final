#!/usr/bin/env bash
# run.sh — Drought prediction pipeline
#
# Usage:
#   bash run.sh                  # 完整執行 Stage 0~5 + 推論
#   bash run.sh --from-stage 3   # 從 Stage 3 開始（特徵快取已建立）
#   bash run.sh --from-stage 4   # 只重訓模型
#   bash run.sh --force          # 忽略所有快取，全部重跑
#   bash run.sh --predict-only   # 只執行推論（models 已存在）
#   bash run.sh --from-stage 4 --force   # 組合使用

set -euo pipefail

# ── 顏色輸出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_stage()   { echo -e "\n${BOLD}${GREEN}━━━ $* ━━━${NC}"; }
log_banner()  {
    echo -e "\n${BOLD}${BLUE}============================================================${NC}"
    echo -e "${BOLD}${BLUE}  Drought Prediction Pipeline${NC}"
    echo -e "${BOLD}${BLUE}============================================================${NC}\n"
}

# ── 參數解析 ──────────────────────────────────────────────────────────────────
FROM_STAGE=0
FORCE=""
PREDICT_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-stage)
            FROM_STAGE="$2"
            shift 2
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        --predict-only)
            PREDICT_ONLY=true
            shift
            ;;
        *)
            log_error "Unknown argument: $1"
            echo "Usage: bash run.sh [--from-stage N] [--force] [--predict-only]"
            exit 1
            ;;
    esac
done

# ── 環境設定 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/code"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="$LOG_DIR/train_${TIMESTAMP}.log"
PREDICT_LOG="$LOG_DIR/predict_${TIMESTAMP}.log"

log_banner

# ── Python 確認 ───────────────────────────────────────────────────────────────
if ! command -v python &>/dev/null; then
    log_error "python not found. Please activate your conda/venv environment first."
    exit 1
fi
PYTHON_VER="$(python --version 2>&1)"
log_info "Python: $PYTHON_VER"
log_info "Code dir: $CODE_DIR"
log_info "Log dir:  $LOG_DIR"

# ── 資料完整性檢查 ────────────────────────────────────────────────────────────
check_data() {
    log_stage "Data Check"
    local missing=0

    for f in \
        "$SCRIPT_DIR/dataset/data/train.csv" \
        "$SCRIPT_DIR/dataset/data/test.csv" \
        "$SCRIPT_DIR/dataset/sample_submission.csv"
    do
        if [[ -f "$f" ]]; then
            local size
            size="$(du -h "$f" | cut -f1)"
            log_ok "Found: $(basename "$f")  ($size)"
        else
            log_error "Missing: $f"
            missing=$((missing + 1))
        fi
    done

    if [[ $missing -gt 0 ]]; then
        log_error "$missing required file(s) missing. Aborting."
        exit 1
    fi
}

# ── Stage 執行函數 ────────────────────────────────────────────────────────────
run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    local t_start t_end elapsed

    log_stage "Stage ${stage_num}: ${stage_name}"
    t_start="$(date +%s)"

    cd "$CODE_DIR"
    if python train.py --from-stage "$stage_num" $FORCE 2>&1 | tee -a "$TRAIN_LOG"; then
        t_end="$(date +%s)"
        elapsed=$((t_end - t_start))
        log_ok "Stage ${stage_num} done in ${elapsed}s"
    else
        log_error "Stage ${stage_num} failed. Check log: $TRAIN_LOG"
        exit 1
    fi
}

run_predict() {
    log_stage "Inference"
    local t_start t_end elapsed
    t_start="$(date +%s)"

    cd "$CODE_DIR"
    if python predict.py 2>&1 | tee -a "$PREDICT_LOG"; then
        t_end="$(date +%s)"
        elapsed=$((t_end - t_start))
        log_ok "Inference done in ${elapsed}s"

        local sub="$SCRIPT_DIR/submission.csv"
        if [[ -f "$sub" ]]; then
            local rows
            rows="$(wc -l < "$sub")"
            log_ok "Submission: $sub  (${rows} lines)"
        fi
    else
        log_error "Inference failed. Check log: $PREDICT_LOG"
        exit 1
    fi
}

# ── 主流程 ────────────────────────────────────────────────────────────────────
T_TOTAL_START="$(date +%s)"

if $PREDICT_ONLY; then
    log_info "Mode: predict-only"
    run_predict
else
    check_data
    log_info "Mode: train  from_stage=${FROM_STAGE}  force=${FORCE:-none}"
    log_info "Train log: $TRAIN_LOG"

    if [[ $FROM_STAGE -le 0 ]]; then
        log_stage "Stage 0: Preprocessing artifacts"
        log_info "Winsorization / Log-sqrt transform / Rank normalization"
        cd "$CODE_DIR"
        python train.py --from-stage 0 $FORCE 2>&1 | tee -a "$TRAIN_LOG" || {
            log_error "Stage 0 failed."
            exit 1
        }
        log_ok "Stage 0 complete"
    fi

    if [[ $FROM_STAGE -le 1 ]]; then
        log_stage "Stage 1: Climatology"
        log_info "Region-level climate baseline statistics"
        cd "$CODE_DIR"
        python train.py --from-stage 1 $FORCE 2>&1 | tee -a "$TRAIN_LOG" || {
            log_error "Stage 1 failed."
            exit 1
        }
        log_ok "Stage 1 complete"
    fi

    if [[ $FROM_STAGE -le 2 ]]; then
        log_stage "Stage 2: Proxy Ridge"
        log_info "Meteorological signals -> drought severity proxy"
        cd "$CODE_DIR"
        python train.py --from-stage 2 $FORCE 2>&1 | tee -a "$TRAIN_LOG" || {
            log_error "Stage 2 failed."
            exit 1
        }
        log_ok "Stage 2 complete"
    fi

    if [[ $FROM_STAGE -le 3 ]]; then
        log_stage "Stage 3: Feature Matrix"
        log_info "Building 405-dim features (parallel, with SHA1 cache)"
        cd "$CODE_DIR"
        python train.py --from-stage 3 $FORCE 2>&1 | tee -a "$TRAIN_LOG" || {
            log_error "Stage 3 failed."
            exit 1
        }
        log_ok "Stage 3 complete"
    fi

    log_stage "Stage 4+5: LightGBM Training + Calibration"
    log_info "Per-horizon walk-forward ensemble (fw=1..5, 3 folds each)"
    cd "$CODE_DIR"
    python train.py --from-stage 4 $FORCE 2>&1 | tee -a "$TRAIN_LOG" || {
        log_error "Stage 4+5 failed."
        exit 1
    }
    log_ok "Stage 4+5 complete"

    run_predict
fi

# ── 總結 ──────────────────────────────────────────────────────────────────────
T_TOTAL_END="$(date +%s)"
TOTAL_ELAPSED=$((T_TOTAL_END - T_TOTAL_START))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo -e "\n${BOLD}${GREEN}============================================================${NC}"
echo -e "${BOLD}${GREEN}  Pipeline Complete  (${TOTAL_MIN}m ${TOTAL_SEC}s)${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
log_info "Train log:   $TRAIN_LOG"
log_info "Predict log: $PREDICT_LOG"
log_info "Submission:  $SCRIPT_DIR/submission.csv"
