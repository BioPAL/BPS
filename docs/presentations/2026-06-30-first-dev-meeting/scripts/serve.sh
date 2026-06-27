#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="$(cd "$ROOT/../.." && pwd)"
STATIC="$DOCS/_extra_static/presentations"
PORT="${1:-8765}"

if [[ ! -f "$STATIC/$(basename "$ROOT")/index.html" ]]; then
  echo "Published deck not found — running make publish…"
  make -C "$ROOT" publish
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Port $PORT already in use — deck may already be served."
  echo "Use another port: bash scripts/serve.sh 8766"
  exit 1
fi

DECK_ID="$(basename "$ROOT")"
echo "Serving $STATIC at http://localhost:$PORT/$DECK_ID/index.html"
cd "$STATIC"
exec python3 -m http.server "$PORT"
