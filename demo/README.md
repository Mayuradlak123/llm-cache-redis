# Demo: Groq + a Tailwind UI

A small web page that puts `llm-cache` in front of the real Groq API, so you can watch a
paraphrased question skip the LLM entirely.

![flow](https://img.shields.io/badge/MISS-calls%20Groq-orange) ![flow](https://img.shields.io/badge/HIT-no%20API%20call-brightgreen)

This folder is **not part of the library**. `llm_cache` never talks to a provider — it takes a
`Callable[[str], str]` and knows nothing about who answers. `groq_llm.py` is just one
implementation of that contract, and the PLAN explicitly keeps provider SDKs, servers and
frontends out of the package itself.

## Setup

```bash
cp .env.example .env     # then paste your key into GROQ_API_KEY
uv run python demo/server.py
```

Open <http://127.0.0.1:8000>.

Get a key at <https://console.groq.com/keys>. No key yet? The server still starts and the
page still works — it shows a banner and answers with a local stand-in, so you can see the caching
behaviour before committing to a key.

## Configuration (`.env`)

Every value the demo uses comes from `.env`, with real environment variables taking precedence:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Your key. Without it the demo uses a local stand-in. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model id sent to the API. |
| `TEMPERATURE` | `0.2` | Sampling temperature — also part of the cache identity. |
| `REDIS_URL` | `redis://localhost:6379` | Where to cache. |
| `SIMILARITY_THRESHOLD` | `0.90` | Cosine similarity required for a HIT. |
| `TTL` | `3600` | Seconds until an entry expires. `none` = never. |
| `NAMESPACE` | `example` | Keeps these entries separate from other apps. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Where the demo server listens. |

`.env` is gitignored. `.env.example` is committed and holds no secrets.

Override a single value for one run:

```bash
SIMILARITY_THRESHOLD=0.95 uv run python demo/server.py
```

## No Redis? It still runs

If `REDIS_URL` cannot be reached, the demo falls back to `memory_store.py`, an in-process
stand-in implementing the handful of commands `RedisStore` actually calls. The footer tells you
which backend is live.

That fallback is a demo convenience, **not a library feature** — the data dies with the process
and is never shared between workers. Force it with `--memory`, and start real Redis with
`docker compose up -d`.

## What to try

1. Ask **"What is Docker?"** → `CACHE MISS`, Groq is called, note the latency.
2. Ask **"Can you explain Docker?"** → `CACHE HIT`, no API call, near-instant.
3. Ask **"How do I make cold brew coffee?"** → `CACHE MISS` (similarity ≈ 0.08).

The counters at the top show requests, hit rate, **actual API calls made** and tokens consumed.
The gap between "Requests" and "LLM calls made" is what the cache saved you.

## Files

| File | Purpose |
| --- | --- |
| `server.py` | Stdlib HTTP server (no FastAPI) exposing `/api/ask`, `/api/state`, `/api/clear`. |
| `index.html` | The UI. Tailwind via CDN, vanilla JS, no build step. |
| `groq_llm.py` | Groq chat-completions adapter built on `urllib`. Adds no dependencies. |
| `config.py` | `.env` loader and typed settings. |
| `memory_store.py` | In-process Redis stand-in for running without Docker. |

## Flags

```bash
uv run python demo/server.py --fake-llm   # never call Groq
uv run python demo/server.py --memory     # skip Redis
```

