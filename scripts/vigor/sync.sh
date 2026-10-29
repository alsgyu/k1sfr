#!/usr/bin/env bash
# Move the non-git parts of VIGOR between the data-prep laptop and the GPU machine.
#
#   scripts/vigor/sync.sh verify-data                               # local data vs git manifest
#   scripts/vigor/sync.sh push-data    <ssh-host> <remote-repo>     # laptop -> GPU: data/vigor/*.npz, verified remotely
#   scripts/vigor/sync.sh pull-results <ssh-host> <remote-repo> [--with-checkpoints]
#                                                                   # GPU -> laptop: exported policies, params, eval/videos
# Code travels through git (branch feat/vigor-fall-safety); datasets are not committed
# (LAFAN1 is CC BY-NC-ND), only their sha256 in vigor/data_manifest.json.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="source/JY_train/JY_train/tasks/manager_based/vigor/data_manifest.json"

check_sums() {  # stdin: "<sha256>  <path>" lines
  python3 -c '
import json, os, sys
m = json.load(open(sys.argv[1]))["datasets"]
seen, bad = set(), 0
for line in sys.stdin:
    digest, path = line.split()
    name = os.path.basename(path)
    if name not in m:
        print(f"  extra    {name}"); continue
    seen.add(name)
    status = "ok" if m[name]["sha256"] == digest else "MISMATCH"
    bad += status != "ok"
    print(f"  {status:8s} {name} {digest[:12]}")
for name in set(m) - seen:
    bad += 1; print(f"  MISSING  {name}")
sys.exit(1 if bad else 0)' "$REPO/$MANIFEST"
}

sum_local() { (cd "$REPO" && for f in data/vigor/*.npz; do [ -e "$f" ] && { shasum -a 256 "$f" 2>/dev/null || sha256sum "$f"; }; done); }

case "${1:-}" in
  verify-data)
    sum_local | check_sums
    ;;
  push-data)
    host="${2:?ssh host}"; remote="${3:?remote repo path}"
    sum_local | check_sums
    ssh "$host" "mkdir -p '$remote/data/vigor'"
    rsync -av --progress "$REPO/data/vigor/" "$host:$remote/data/vigor/"
    echo "remote check:"
    ssh "$host" "cd '$remote' && sha256sum data/vigor/*.npz" | check_sums
    ;;
  pull-results)
    host="${2:?ssh host}"; remote="${3:?remote repo path}"
    filters=(--include='*/' --include='exported/***' --include='params/***' --include='videos/***'
             --include='*.json' --include='git/***' --include='events.out.tfevents.*')
    [ "${4:-}" = "--with-checkpoints" ] && filters+=(--include='model_*.pt')
    mkdir -p "$REPO/results"
    rsync -av --prune-empty-dirs "${filters[@]}" --exclude='*' "$host:$remote/logs/" "$REPO/results/logs/"
    echo "pulled into results/logs (git-ignored)"
    ;;
  *)
    sed -n '2,9p' "$0"; exit 1
    ;;
esac
