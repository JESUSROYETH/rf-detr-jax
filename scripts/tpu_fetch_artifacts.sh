#!/usr/bin/env bash
# Copy the TPU node's measurement JSONs into ignored local artifacts.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[ -f "$here/tpu.local.sh" ] && . "$here/tpu.local.sh"
repo="${TPU_REMOTE_REPO:-rf-detr-jax}"
dest="$here/../artifacts/tpu-measurements"
mkdir -p "$dest"
for name in "$@"; do
  "$here/tpu.sh" pull "$repo/artifacts/$name" "$dest/$name" || echo "missing: $name" >&2
done
ls -la "$dest"
