#!/usr/bin/env bash
set -euo pipefail

# Run in an existing FlagTree 3.8.x IFWA CModel environment.
cd "$(dirname "$0")/.."
export GEMS_VENDOR=intellifusion
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

status=0
for op in abs add sum; do
  python - "$op" <<'PY' || status=1
import sys

import pytest
import torch
import torch_ifwa  # noqa: F401
import triton

import flag_gems

op = sys.argv[1]
target = triton.runtime.driver.active.get_current_target()
if (
    target.backend != "ifwa"
    or flag_gems.vendor_name != "intellifusion"
    or torch.empty(1, device=flag_gems.device).device.type != "ifwa"
):
    raise SystemExit(f"Unexpected FlagGems device or Triton target: {target}")

launches = []

def record_launch(metadata):
    launches.append(metadata)

triton.knobs.runtime.launch_exit_hook.add(record_launch)
code = pytest.main([
    f"tests/test_{op}.py", "-m", op, "--quick", "--ref", "cpu",
    "--record", "json", "--output", f"ifwa-{op}.json",
    f"--junitxml=ifwa-{op}.xml",
])
print(f"{op}: target={target.backend}, completed Triton launches={len(launches)}")
raise SystemExit(code or (0 if launches else 1))
PY
done
exit "$status"
