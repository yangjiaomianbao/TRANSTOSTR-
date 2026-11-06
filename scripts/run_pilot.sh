#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
# A real completed smoke run is required; never infer success from a directory alone.
SMOKE_RUN="${1:?Usage: run_pilot.sh completed-smoke-run [data-path] [run-path] [device] [seed]}"
"$PYTHON" - "$SMOKE_RUN" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
assert json.loads((p/'execution_status.json').read_text())['status']=='smoke_complete'
assert json.loads((p/'validation.json').read_text())['status']=='passed'
assert json.loads((p/'probe/metrics_summary.json').read_text())['experiment']=='smoke'
assert 'passed' in (p/'pytest.log').read_text()
assert (p/'best.pt').is_file() and (p/'node_only/best.pt').is_file()
PY
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
DATA_PATH="${2:-data/pilot_${STAMP}}"
SEED="${5:-42}"
RUN_PATH="${3:-runs/pilot_${STAMP}_seed${SEED}}"
DEVICE="${4:-cuda:0}"
"$PYTHON" - "$DEVICE" <<'PY'
import sys
from src.utils import get_device
print('Pilot device:',get_device(sys.argv[1]))
PY
if [[ -e "$DATA_PATH/manifest.json" ]]; then
  echo "Reusing explicitly selected pilot dataset: $DATA_PATH"
else
  "$PYTHON" -m src.data.generate --config configs/pilot.yaml --out "$DATA_PATH"
fi
VALIDATION_PATH="$(mktemp)"
"$PYTHON" -m src.data.validate --data "$DATA_PATH" | tee "$VALIDATION_PATH"
"$PYTHON" -m src.train --config configs/pilot.yaml --data "$DATA_PATH" --out "$RUN_PATH" --device "$DEVICE" --seed "$SEED"
cp "$VALIDATION_PATH" "$RUN_PATH/validation.json"
rm "$VALIDATION_PATH"
"$PYTHON" -m src.probe --config configs/pilot.yaml --data "$DATA_PATH" --checkpoint "$RUN_PATH/best.pt" --out "$RUN_PATH/probe" --device "$DEVICE"
"$PYTHON" -m pytest -q | tee "$RUN_PATH/pytest.log"
"$PYTHON" - "$RUN_PATH" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
summary=json.loads((p/'probe/metrics_summary.json').read_text())
(p/'execution_status.json').write_text(json.dumps(dict(status='pilot_complete',run_id=summary['training_run_id'],
    probe_run_id=summary['run_id'],tests='passed'),indent=2))
PY
echo "PILOT_SUCCESS=$RUN_PATH"
