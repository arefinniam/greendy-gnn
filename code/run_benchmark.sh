#!/usr/bin/env bash
###############################################################################
# run_benchmark.sh — GreenDyGNN artifact benchmark driver.
#
# Runs 4 methods (default_dgl, bgl, rapidgnn, greendygnn) x 3 datasets x 3
# batch sizes = 36 configurations, each for 30 epochs. Identical congestion is
# applied to every method when --with-congestion is set.
#
# Usage:
#   DATASET_ROOT=/path/to/partitioned/datasets  ./run_benchmark.sh [options]
#
# Options:
#   --with-congestion     Apply time-varying 15-25 ms delays (default off)
#   --methods LIST        Comma-separated subset of methods (default all 4)
#   --datasets LIST       Comma-separated subset of datasets
#   --batch-sizes LIST    Comma-separated subset of batch sizes
#   --epochs N            Epochs per run (default 30)
#   --out DIR             Where to write per-run logs (default ./logs/<ts>)
#   --iface NAME          NIC for tc netem (default eno1)
#
# Required env:
#   DATASET_ROOT   Root path containing:
#     $DATASET_ROOT/OGBN-Products/data/ogbn-products.json
#     $DATASET_ROOT/Reddit/data/reddit.json
#     $DATASET_ROOT/OGBN-Papers/data/ogbn-papers100M.json
###############################################################################
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IP_CONFIG="$DIR/ip_config.txt"
: "${DATASET_ROOT:?DATASET_ROOT must be set (directory with partitioned datasets)}"

# ── Defaults ──
METHODS_DEFAULT="default_dgl,bgl,rapidgnn,greendygnn"
DATASETS_DEFAULT="ogbn-products,reddit,ogbn-papers100M"
BATCHES_DEFAULT="1000,2000,3000"
WITH_CONGESTION=0
NUM_EPOCHS=30
IFACE="eno1"
TIMEOUT_PER_RUN=1800
OUT_DIR=""

METHODS="$METHODS_DEFAULT"
DATASETS="$DATASETS_DEFAULT"
BATCHES="$BATCHES_DEFAULT"

# ── CLI parsing ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-congestion) WITH_CONGESTION=1; shift ;;
        --methods)         METHODS="$2"; shift 2 ;;
        --datasets)        DATASETS="$2"; shift 2 ;;
        --batch-sizes)     BATCHES="$2"; shift 2 ;;
        --epochs)          NUM_EPOCHS="$2"; shift 2 ;;
        --out)             OUT_DIR="$2"; shift 2 ;;
        --iface)           IFACE="$2"; shift 2 ;;
        -h|--help)         sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$DIR/logs/benchmark_$TIMESTAMP}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.log"
: > "$SUMMARY"

# ── Dataset metadata ──
declare -A PART_CONFIGS NCLASSES
PART_CONFIGS["ogbn-products"]="$DATASET_ROOT/OGBN-Products/data/ogbn-products.json"
PART_CONFIGS["reddit"]="$DATASET_ROOT/Reddit/data/reddit.json"
PART_CONFIGS["ogbn-papers100M"]="$DATASET_ROOT/OGBN-Papers/data/ogbn-papers100M.json"
NCLASSES["ogbn-products"]=47
NCLASSES["reddit"]=41
NCLASSES["ogbn-papers100M"]=172

TRAIN_COMMON="--num_gpus 1 --num_hidden 16 --num_layers 2 --fan_out 10,25 \
    --lr 0.003 --dropout 0.5 --log_every 20"

mapfile -t IPS < <(sed 's/[[:space:]]//g' "$IP_CONFIG" | grep .)
# Non-coordinator nodes receive congestion (first IP in ip_config.txt is the
# coordinator/server).
CONG_IPS_FILE="$OUT_DIR/cong_ips.txt"
printf '%s\n' "${IPS[@]:1}" > "$CONG_IPS_FILE"

# ── Cluster cleanup primitives ──
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5)
ssh_node() { ssh "${SSH_OPTS[@]}" "$1" "$2" 2>/dev/null || true; }

kill_all_processes() {
    pkill -TERM -f "python3.*(train_|launch)" 2>/dev/null || true
    pkill -TERM -f "torch.distributed" 2>/dev/null || true
    pkill -TERM -f "dgl.distributed" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "python3.*(train_|launch)" 2>/dev/null || true
    pkill -KILL -f "torch.distributed" 2>/dev/null || true
    pkill -KILL -f "dgl.distributed" 2>/dev/null || true
    for p in 30050 30051 30052 30053 29500 29501; do
        fuser -k -TERM ${p}/tcp 2>/dev/null || true
    done

    local REMOTE_KILL='
        pkill -TERM -f "python3.*(train_|launch)" 2>/dev/null || true
        pkill -TERM -f "torch.distributed" 2>/dev/null || true
        pkill -TERM -f "dgl.distributed" 2>/dev/null || true
        sleep 2
        pkill -KILL -f "python3.*(train_|launch)" 2>/dev/null || true
        pkill -KILL -f "torch.distributed" 2>/dev/null || true
        pkill -KILL -f "dgl.distributed" 2>/dev/null || true
        for p in 30050 30051 30052 30053 29500 29501; do
            fuser -k -TERM ${p}/tcp 2>/dev/null || true
        done
    '
    for ip in "${IPS[@]}"; do ssh_node "$ip" "$REMOTE_KILL"; done
}

clear_congestion() {
    python3 "$DIR/congestion.py" --cleanup --cong_ips "$CONG_IPS_FILE" \
        --iface "$IFACE" >/dev/null 2>&1 || true
}

full_cleanup() {
    kill_all_processes
    clear_congestion
    sleep 3
}

# ── Build per-method training command ──
build_command() {
    local method="$1" dataset="$2" batch="$3" out="$4"
    local pc="${PART_CONFIGS[$dataset]}"
    local nc="${NCLASSES[$dataset]}"
    local common="--graph_name $dataset --ip_config $IP_CONFIG --part_config $pc \
        --num_epochs $NUM_EPOCHS --batch_size $batch --out_dir $out $TRAIN_COMMON"

    case "$method" in
        default_dgl) echo "python3 $DIR/train_default.py   $common --n_classes $nc" ;;
        bgl)         echo "python3 $DIR/train_bgl.py       $common --n_classes $nc" ;;
        rapidgnn)    echo "python3 $DIR/train_rapidgnn.py  $common --window_size 16 --cache_size 100000" ;;
        greendygnn)  echo "python3 $DIR/train_greendygnn.py $common --window_size 16 --cache_size 100000" ;;
        *) echo "ERROR: unknown method $method" >&2; return 1 ;;
    esac
}

# ── Run one configuration ──
run_one() {
    local n="$1" total="$2" method="$3" dataset="$4" batch="$5"
    local label="${method}/${dataset}/B${batch}"
    local run_dir="$OUT_DIR/$method/$dataset/B${batch}"
    local log="$run_dir/run.log"
    mkdir -p "$run_dir"

    echo ""
    echo "── Run $n/$total : $label ── $(date '+%H:%M:%S')"
    full_cleanup

    local cmd
    cmd=$(build_command "$method" "$dataset" "$batch" "$run_dir") || return 1
    local t0; t0=$(date +%s)

    timeout "$TIMEOUT_PER_RUN" python3 "$DIR/launch.py" \
        --workspace "$DIR" \
        --num_trainers 1 --num_servers 1 \
        --part_config "${PART_CONFIGS[$dataset]}" \
        --ip_config "$IP_CONFIG" \
        "$cmd" > "$log" 2>&1 &
    local train_pid=$!

    local cong_pid=""
    if [[ $WITH_CONGESTION -eq 1 ]]; then
        sleep 3
        python3 "$DIR/congestion.py" \
            --log_file "$log" \
            --total_epochs "$NUM_EPOCHS" \
            --cong_ips "$CONG_IPS_FILE" \
            --iface "$IFACE" \
            --timeout $((TIMEOUT_PER_RUN + 300)) \
            > "$run_dir/congestion.log" 2>&1 &
        cong_pid=$!
    fi

    local exit_code=0
    wait "$train_pid" || exit_code=$?
    [[ -n "$cong_pid" ]] && { kill "$cong_pid" 2>/dev/null || true; wait "$cong_pid" 2>/dev/null || true; }

    full_cleanup
    python3 "$DIR/parse_results.py" \
        --log_file "$log" --output "$run_dir/metrics.json" \
        --method "$method" --dataset "$dataset" --batch_size "$batch" \
        2>/dev/null || true

    local duration=$(( $(date +%s) - t0 ))
    local status="OK"
    [[ $exit_code -ne 0 ]] && status="FAIL(exit=$exit_code)"
    [[ $exit_code -eq 124 ]] && status="TIMEOUT"
    echo "  ── $status | ${duration}s ──"
    echo "$n/$total|$label|$status|${duration}s" >> "$SUMMARY"
}

# ── Main ──
IFS=',' read -ra M_ARR <<< "$METHODS"
IFS=',' read -ra D_ARR <<< "$DATASETS"
IFS=',' read -ra B_ARR <<< "$BATCHES"
TOTAL=$(( ${#M_ARR[@]} * ${#D_ARR[@]} * ${#B_ARR[@]} ))

echo "═══════════════════════════════════════════════════════════════════"
echo " GreenDyGNN benchmark"
echo "  methods      : ${METHODS}"
echo "  datasets     : ${DATASETS}"
echo "  batch sizes  : ${BATCHES}"
echo "  epochs       : $NUM_EPOCHS"
echo "  congestion   : $([[ $WITH_CONGESTION -eq 1 ]] && echo 'yes (15-25 ms)' || echo 'clean')"
echo "  results      : $OUT_DIR"
echo "═══════════════════════════════════════════════════════════════════"

full_cleanup

N=0
for dataset in "${D_ARR[@]}"; do
    for batch in "${B_ARR[@]}"; do
        for method in "${M_ARR[@]}"; do
            N=$((N + 1))
            run_one "$N" "$TOTAL" "$method" "$dataset" "$batch"
        done
    done
done

full_cleanup
echo ""
echo "═══ Benchmark complete ═══"
echo "Summary: $SUMMARY"
column -t -s'|' "$SUMMARY"
