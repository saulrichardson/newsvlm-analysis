# Gateway

Lightweight FastAPI gateway that exposes a single Responses-style endpoint and forwards chat calls to OpenAI, Gemini, Claude, or a built-in Echo provider. It streams SSE, tags requests with trace IDs, and offers a tiny in-memory agent message bus. An optional EDGAR module can batch SEC document jobs through OpenAI.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # or use poetry
pip install -e .
cp .env.example .env   # add your OPENAI_KEY / GEMINI_KEY / CLAUDE_KEY
make serve             # runs uvicorn with reload
```

Key endpoints:
- `GET /healthz` – liveness + available providers
- `GET /readyz` – readiness (checks OpenAI key/reachability)
- `POST /v1/responses` – OpenAI Responses-compatible SSE stream
- `POST /v1/agents/messages` / `GET /v1/agents/{agent_id}/messages` – in-memory agent bus

## Calling the gateway

Prefix the model with a provider (or set `DEFAULT_PROVIDER`):
```bash
curl -N -X POST http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini:gemini-3-pro-preview","input":[{"role":"user","content":"hi"}],"stream":true}'
```

## Provider adapters

- `openai` → Responses API at `https://api.openai.com/v1/responses`
- `gemini` → `generateContent` on `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- `claude` → `https://api.anthropic.com/v1/messages`
- `echo` → local deterministic echo for tests

Missing keys raise `provider_not_configured`; let it fail fast.

## Development

```bash
make test         # pytest
make lint         # ruff
make format       # black + ruff --fix
make type-check   # mypy
```

Docker: `make docker-build && make start` (compose file in `docker/`).

## Agent helper

`gateway.client.GatewayAgentClient` wraps the SSE stream so agents can consume responses easily; `complete_response_sync` provides a blocking helper for quick scripts.  Use `build_user_message(prompt, image_paths=[...])` or `build_user_message(prompt, image_bytes=[...])` to attach images.

## EDGAR pipeline (optional)

`gateway.edgar` exposes `/jobs` to run SEC segment prompts through OpenAI. Requires `edgar_filing_pipeline` deps; avoid if that stack isn’t installed.
