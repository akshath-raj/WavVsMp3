#!/usr/bin/env bash
# Fetch the CREMA-D AudioWAV corpus.
#
# The upstream repo stores audio in git-lfs; rather than requiring a git-lfs
# install we pull each object from GitHub's LFS media endpoint, which serves the
# real file contents. Idempotent: files already present with non-zero size are
# skipped, so re-running resumes a partial download.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/raw"
LIST="${1:-$ROOT/data/filelist.txt}"
BASE="https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"
JOBS="${JOBS:-12}"

mkdir -p "$OUT"

fetch_one() {
  local name="$1"
  local dest="$OUT/$name"
  if [[ -s "$dest" ]]; then return 0; fi
  curl -sSL --fail --retry 5 --retry-delay 2 --retry-all-errors -m 120 \
       -o "$dest.part" "$BASE/$name" && mv "$dest.part" "$dest"
}
export -f fetch_one
export OUT BASE

xargs -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {} < "$LIST"

echo "downloaded: $(find "$OUT" -name '*.wav' -size +0 | wc -l) / $(wc -l < "$LIST")"
