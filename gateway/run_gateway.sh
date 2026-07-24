#!/usr/bin/env bash
# Start the OpenAI-compatible gateway (public entry point) on port 9000.
# Reads the API key from gateway/.env (created on first run if missing).
# Start serve_backends.sh FIRST, then this in a second tmux pane.
set -uo pipefail

PY="/home/work/.keural_venv/bin/python"
DIR="/home/work/keural MoE Pro/gateway"
export PYTHONPATH=

# create a key on first run
if [ ! -f "$DIR/.env" ]; then
  KEY="keural-$("$PY" -c 'import secrets; print(secrets.token_hex(20))')"
  echo "KEURAL_GATEWAY_KEY=$KEY" > "$DIR/.env"
  echo "Generated a new gateway API key -> $DIR/.env"
fi
set -a; . "$DIR/.env"; set +a
export KEURAL_GATEWAY_PORT="${KEURAL_GATEWAY_PORT:-9000}"

echo "Gateway key: $KEURAL_GATEWAY_KEY"
echo "Gateway URL: http://0.0.0.0:$KEURAL_GATEWAY_PORT/v1"
exec "$PY" "$DIR/gateway.py"
