# Keural Nova — Model Gateway (for integration)

A single **OpenAI-compatible** API that serves both models. Any AI platform or SDK that speaks
the OpenAI API can use it — just set the base URL, the API key, and the model name.

## Connection details

| Field | Value |
|---|---|
| **Base URL** | `http://<HOST>:9000/v1` |
| **API key** | see `gateway/.env` on the server (`KEURAL_GATEWAY_KEY=...`) |
| **Models** | `keural-nova-v1.0` , `keural-nova-v1.1` |

`<HOST>` = the server's address reachable from your machine (see **Networking** below).

## Quick test (curl)

```bash
curl http://<HOST>:9000/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "keural-nova-v1.0",
        "messages": [{"role":"user","content":"한국의 수도는 어디인가요?"}],
        "temperature": 0.7
      }'
```

List models / health:
```bash
curl -H "Authorization: Bearer <API_KEY>" http://<HOST>:9000/v1/models
curl http://<HOST>:9000/health      # no key needed; shows backend up/down
```

## Python (openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://<HOST>:9000/v1", api_key="<API_KEY>")
r = client.chat.completions.create(
    model="keural-nova-v1.1",
    messages=[{"role": "user", "content": "가을에 어울리는 짧은 시를 지어줘."}],
)
print(r.choices[0].message.content)
```

## Plugging into an AI platform

Add it as an **OpenAI-compatible / custom OpenAI provider**:

- **OpenWebUI:** Settings → Connections → OpenAI API → URL `http://<HOST>:9000/v1`, key `<API_KEY>`.
- **Dify:** Model Provider → OpenAI-API-compatible → API endpoint `http://<HOST>:9000/v1`,
  model name `keural-nova-v1.0`.
- **LibreChat / LiteLLM / anything else:** same three fields (base URL, key, model name).

Both streaming and non-streaming are supported; the gateway defaults each request to the
model's trained non-thinking chat format automatically.

## Public access (share with anyone)

The gateway is built to work **through the KT Cloud proxy**, so anyone on the internet with the
URL + key can use it — no shared network needed. The gateway tolerates the proxy's path prefix,
so both of these work identically:

- direct:  `http://<server-ip>:9000/v1`
- proxied: `https://proxy2.nipa2025.ktcloud.com:<EXTERNAL_PORT>/proxy/9000/v1`

**To get the public URL:** open `.../proxy/9000/health` in your browser the same way you opened
`/proxy/7861` before — the `<EXTERNAL_PORT>` in the address bar (KT Cloud assigns it, e.g.
10250/10257) plus `/proxy/9000` is your public base. If `/health` returns
`{"gateway":"ok",...}`, the public URL works; share `<that base>/v1` + the key with anyone.

- **Same private network:** `http://<server-ip>:9000/v1` also works directly.
- **SSH users:** `ssh -L 9000:localhost:9000 <server>` → `http://localhost:9000/v1`.

If the proxied `/health` does NOT return JSON, paste me the exact browser URL and I'll adjust.

## Security notes

- The two model backends run on **localhost only** (127.0.0.1:8001 / :8002) and are **not**
  reachable from outside — the gateway is the sole entry point, and every request needs the key.
- Treat `KEURAL_GATEWAY_KEY` like a password. To rotate it: edit `gateway/.env` and restart
  `run_gateway.sh`. To issue different keys per user, tell me and I'll add multi-key support.

## Running it (server side, in tmux)

```bash
# pane 1 — model backends (wait for BOTH "Application startup complete")
cd "/home/work/keural MoE Pro/gateway" && bash serve_backends.sh

# pane 2 — the gateway
cd "/home/work/keural MoE Pro/gateway" && bash run_gateway.sh
```

Files: `serve_backends.sh` (two localhost vLLM servers), `gateway.py` (the OpenAI-compatible
router), `run_gateway.sh` (launches the gateway + manages the key in `.env`).
