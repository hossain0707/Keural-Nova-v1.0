#!/usr/bin/env bash
# Make the Keural Nova gateway PUBLIC from this box — works on any network,
# no KT Cloud proxy, no firewall changes.
#
# Run this in a tmux pane (NOT via automation — a tunnel needs a real terminal).
# It opens a Cloudflare tunnel to the local gateway (port 9000) and prints a
# public https URL that anyone can use with the API key.
#
# Prereq: the gateway must already be up (serve_backends.sh + run_gateway.sh).
set -uo pipefail
DIR="/home/work/keural MoE Pro/gateway"

# 1) gateway must be running locally
if ! curl -s -m 5 http://localhost:9000/health >/dev/null 2>&1; then
  echo "!! Gateway not responding on :9000."
  echo "   Start it first (two other tmux panes):"
  echo "     bash \"$DIR/serve_backends.sh\"   # wait for BOTH 'Application startup complete'"
  echo "     bash \"$DIR/run_gateway.sh\""
  exit 1
fi

# 2) read the API key
KEY=$(grep -oE 'keural-[0-9a-f]+' "$DIR/.env" 2>/dev/null | head -1)
[ -z "$KEY" ] && { echo "!! No API key in $DIR/.env — run run_gateway.sh once to create it."; exit 1; }

echo "Gateway healthy. Opening public tunnel (HTTP/2, works through restrictive networks)..."
echo "Keep this pane open — the tunnel lives as long as this runs."
echo

# 3) run the tunnel in the foreground; print a SHARE block when the URL appears
"$DIR/bin/cloudflared" tunnel --no-autoupdate --protocol http2 --url http://localhost:9000 2>&1 | \
while IFS= read -r line; do
  echo "$line"
  if echo "$line" | grep -qoE 'https://[a-z0-9-]+\.trycloudflare\.com'; then
    URL=$(echo "$line" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
    cat <<BANNER

============================  SHARE WITH ANYONE  ============================
  Base URL : $URL/v1
  API key  : $KEY
  Models   : keural-nova-v1.0 , keural-nova-v1.1

  Quick test (from any machine, any network):
    curl -s $URL/v1/chat/completions \\
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \\
      -d '{"model":"keural-nova-v1.0","messages":[{"role":"user","content":"안녕하세요"}]}'
=============================================================================

BANNER
  fi
done
