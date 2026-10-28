#!/usr/bin/env bash
# Thin wrapper around gcloud for a locally configured TPU node.
#
# The node's identity is not in this repository. Put it in scripts/tpu.local.sh, which is
# gitignored, or export the four variables yourself:
#
#   TPU_NAME=<tpu-name> TPU_ZONE=<zone> \
#   TPU_ACCOUNT=you@example.com TPU_PROJECT=your-project ./scripts/tpu.sh ssh 'hostname'
#
# Optional: TPU_REMOTE_REPO (default rf-detr-jax) and TPU_REMOTE_PYTHON (default python3)
# name the repository copy and interpreter on the node for tpurun.sh and tpu_fetch_artifacts.sh.
# The account and project are passed on every gcloud command so the active configuration is
# never assumed.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[ -f "$here/tpu.local.sh" ] && . "$here/tpu.local.sh"

for name in TPU_NAME TPU_ZONE TPU_ACCOUNT TPU_PROJECT; do
  if [ -z "${!name:-}" ]; then
    echo "$name is not set; see the comment at the top of $0" >&2
    exit 2
  fi
done
FLAGS=(--zone="$TPU_ZONE" --account="$TPU_ACCOUNT" --project="$TPU_PROJECT")

case "${1:-}" in
  ssh)   shift; exec gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${FLAGS[@]}" --worker=all --command="$*" ;;
  push)  shift; dst="${!#}"; set -- "${@:1:$#-1}"
         exec gcloud compute tpus tpu-vm scp "${FLAGS[@]}" --worker=all --recurse "$@" "$TPU_NAME:$dst" ;;
  pull)  shift; exec gcloud compute tpus tpu-vm scp "${FLAGS[@]}" --worker=0 --recurse "$TPU_NAME:$1" "$2" ;;
  *) echo "usage: tpu.sh {ssh <cmd>|push <local...> <remote>|pull <remote> <local>}" >&2; exit 2 ;;
esac
