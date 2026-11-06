#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
DATA_PATH="${1:-data/smoke_${STAMP}}"
RUN_PATH="${2:-runs/smoke_${STAMP}_seed42}"
"$PYTHON" -m pytest -q
if [[ -e "$DATA_PATH/manifest.json" ]]; then
  echo "Reusing explicitly selected dataset after full hash/physics audit: $DATA_PATH"
else
  "$PYTHON" -m src.data.generate --config configs/smoke.yaml --out "$DATA_PATH"
fi
VALIDATION_PATH="$(mktemp)"
"$PYTHON" -m src.data.validate --data "$DATA_PATH" | tee "$VALIDATION_PATH"
"$PYTHON" -m src.train --config configs/smoke.yaml --data "$DATA_PATH" --out "$RUN_PATH" --device cpu --seed 42
cp "$VALIDATION_PATH" "$RUN_PATH/validation.json"
rm "$VALIDATION_PATH"
"$PYTHON" -m src.probe --config configs/smoke.yaml --data "$DATA_PATH" --checkpoint "$RUN_PATH/best.pt" --out "$RUN_PATH/probe" --device cpu
"$PYTHON" -m pytest -q | tee "$RUN_PATH/pytest.log"
"$PYTHON" -m src.report --probe "$RUN_PATH/probe" --data "$DATA_PATH" --out "reports/$(basename "$RUN_PATH").md"
"$PYTHON" - "$RUN_PATH" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
summary=json.loads((p/'probe/metrics_summary.json').read_text())
(p/'execution_status.json').write_text(json.dumps(dict(status='smoke_complete',run_id=summary['training_run_id'],
    probe_run_id=summary['run_id'],tests='passed',report='reports/'+p.name+'.md'),indent=2))
PY
echo "SMOKE_SUCCESS=$RUN_PATH"
