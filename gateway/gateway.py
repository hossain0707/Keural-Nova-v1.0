"""Keural Nova — OpenAI-compatible API gateway.

A single OpenAI-compatible endpoint that fronts both model backends:
  keural-nova-v1.0  -> http://127.0.0.1:8001  (localhost-only vLLM, GPU 0)
  keural-nova-v1.1  -> http://127.0.0.1:8002  (localhost-only vLLM, GPU 1)

The backends are bound to localhost and are NOT reachable from outside. Only this
gateway is exposed, and every request must carry a valid API key. Any tool that
speaks the OpenAI API (OpenWebUI, Dify, LibreChat, LiteLLM, the openai SDK, curl)
points at this one URL.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,
POST /v1/completions — streaming and non-streaming both supported.

Config via env:
  KEURAL_GATEWAY_KEY   required API key clients must send (Bearer or ?api_key)
  KEURAL_GATEWAY_PORT  default 9000
"""
from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

BACKENDS = {
    "keural-nova-v1.0": "http://127.0.0.1:8001",
    "keural-nova-v1.1": "http://127.0.0.1:8002",
}
DEFAULT_MODEL = "keural-nova-v1.0"

GATEWAY_KEY = os.environ.get("KEURAL_GATEWAY_KEY", "")
PORT = int(os.environ.get("KEURAL_GATEWAY_PORT", "9000"))

app = FastAPI(title="Keural Nova Gateway", version="1.0")
# generous timeout: first token can take a moment under load; no read cap for streams
_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))


@app.middleware("http")
async def strip_proxy_prefix(request: Request, call_next):
    """Make the gateway work behind a path-based reverse proxy (e.g. KT Cloud's
    /proxy/9000/... form) as well as directly. If the incoming path contains a
    '/proxy/<port>/' segment, rewrite it to the bare '/v1/...' the router expects,
    so clients can use either the direct URL or the proxied public URL unchanged."""
    path = request.scope.get("path", "")
    marker = "/proxy/"
    if marker in path:
        rest = path.split(marker, 1)[1]           # "9000/v1/chat/completions"
        seg = rest.split("/", 1)
        new_path = "/" + (seg[1] if len(seg) > 1 else "")
        request.scope["path"] = new_path
        request.scope["raw_path"] = new_path.encode()
    return await call_next(request)


def _check_key(authorization: str | None, api_key_q: str | None) -> None:
    if not GATEWAY_KEY:  # misconfigured server — fail closed
        raise HTTPException(500, "gateway API key not configured")
    supplied = None
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    supplied = supplied or api_key_q
    if supplied != GATEWAY_KEY:
        raise HTTPException(401, "invalid or missing API key")


def _resolve_backend(model: str | None) -> tuple[str, str]:
    """Return (backend_base_url, resolved_model_name)."""
    name = model or DEFAULT_MODEL
    # tolerate common aliases (v1, 1.0, with/without prefix)
    alias = name.strip().lower().replace("_", "-")
    for canonical in BACKENDS:
        if alias == canonical or alias.endswith(canonical) or canonical.endswith(alias):
            return BACKENDS[canonical], canonical
    raise HTTPException(404, f"unknown model '{model}'. Available: {list(BACKENDS)}")


@app.get("/health")
async def health():
    status = {}
    for name, url in BACKENDS.items():
        try:
            r = await _client.get(f"{url}/health", timeout=5)
            status[name] = "up" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            status[name] = f"down ({type(e).__name__})"
    return {"gateway": "ok", "backends": status}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(None), api_key: str | None = None):
    _check_key(authorization, api_key)
    return {"object": "list",
            "data": [{"id": n, "object": "model", "owned_by": "mkd"} for n in BACKENDS]}


async def _proxy(path: str, request: Request, authorization: str | None, api_key_q: str | None):
    _check_key(authorization, api_key_q)
    body = await request.json()
    backend, resolved = _resolve_backend(body.get("model"))
    body["model"] = resolved
    # default to the model's trained non-thinking path unless caller overrides
    body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    url = f"{backend}{path}"

    if body.get("stream"):
        async def event_stream():
            async with _client.stream("POST", url, json=body) as r:
                if r.status_code != 200:
                    detail = await r.aread()
                    yield f"data: {{\"error\": {detail.decode(errors='replace')!r}}}\n\n"
                    return
                async for chunk in r.aiter_raw():
                    yield chunk
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    r = await _client.post(url, json=body)
    return JSONResponse(status_code=r.status_code, content=r.json())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(None),
                           api_key: str | None = None):
    return await _proxy("/v1/chat/completions", request, authorization, api_key)


@app.post("/v1/completions")
async def completions(request: Request, authorization: str | None = Header(None),
                      api_key: str | None = None):
    return await _proxy("/v1/completions", request, authorization, api_key)


if __name__ == "__main__":
    import uvicorn
    if not GATEWAY_KEY:
        raise SystemExit("Set KEURAL_GATEWAY_KEY before starting the gateway.")
    print(f"Keural Nova gateway on 0.0.0.0:{PORT} — models: {list(BACKENDS)}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
