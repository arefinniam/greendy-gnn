#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/greendygnn_smoke.XXXXXX")"
LOG_FILE="$OUT_DIR/run.log"
METRICS_FILE="$OUT_DIR/metrics.json"
FIG_DIR="$OUT_DIR/figures"

python3 - <<'PY'
import importlib

for name in ("torch", "dgl", "numpy", "pynvml", "matplotlib"):
    importlib.import_module(name)
print("dependency imports ok")
PY

cat > "$LOG_FILE" <<'LOG'
Part 0 Ep00: 1.23s GPU=4.5J loss=0.1 acc=0.9
Part 0: Total GPU energy consumed: 10.00J
Part 0: Total CPU energy consumed: 2.00J
Part 0: Total energy consumed: 12.00J
LOG

python3 "$ROOT/code/parse_results.py" \
    --log_file "$LOG_FILE" \
    --output "$METRICS_FILE" \
    --method smoke \
    --dataset tiny \
    --batch_size 1

python3 "$ROOT/code/gen_figures.py" \
    --data "$ROOT/data/paper_data.json" \
    --out_dir "$FIG_DIR"

test -s "$METRICS_FILE"
test "$(find "$FIG_DIR" -name '*.pdf' | wc -l | tr -d ' ')" = "8"
echo "smoke test ok: $OUT_DIR"
