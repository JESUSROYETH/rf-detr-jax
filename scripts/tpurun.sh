#!/usr/bin/env bash
# Start one TPU job detached on the node and return immediately.
# Usage: tpurun.sh <log-name> <script.py> [args...]
# The repository copy and interpreter on the node come from TPU_REMOTE_REPO and
# TPU_REMOTE_PYTHON (see tpu.sh), defaulting to ~/rf-detr-jax and python3.
set -euo pipefail
log="$1"; shift
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[ -f "$here/tpu.local.sh" ] && . "$here/tpu.local.sh"
repo="${TPU_REMOTE_REPO:-rf-detr-jax}"
python="${TPU_REMOTE_PYTHON:-python3}"
"$here/tpu.sh" ssh "cd ~/$repo && rm -f ~/$log.log && setsid ./scripts/tpuenv.sh $python $* > ~/$log.log 2>&1 < /dev/null & disown; sleep 2; test -s ~/$log.log || pgrep -f '$1' >/dev/null; echo started-$log"
