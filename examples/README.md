# Customer integration examples

What a customer's codebase looks like before and after integrating the
inferlog SDK. The files aren't runnable scripts; they're reference
snippets to read top-to-bottom or paste into a slide.

The SDK itself lives in `sdk/` — that's Ollive code. The examples here
are written from the customer's perspective: their `gateway/`, their
`api/`, their `worker/` — whichever service is making LLM calls.

| File | What it shows |
|---|---|
| `custom_http_provider.py` | The customer has their own LLM behind an internal URL (not OpenAI/Anthropic/Ollama). One `ProviderHandler` class + two startup lines. Customer's call site is unchanged. |
| `in_process_provider.py` | The customer's "model" is a Python object — embedded inference, an in-house judge, or a mock. `inferlog.wrap_provider(name=MyProvider())` at startup; call site changes from `model.complete(...)` to `client.complete(provider="my_model", ...)`. |

For the **built-in** providers — OpenAI, Anthropic, Ollama,
OpenAI-compatible proxies (vLLM / OpenRouter / Together / LiteLLM
proxy) — the customer adds nothing beyond `inferlog.init(...)` at
startup. The HTTP traffic these vendor SDKs produce is captured
automatically by URL pattern. See `sdk/README.md` for that one-line
setup; see `gateway/app/llm.py` in this repo for a live demo of all
three integration modes (global capture, per-client transport,
in-process wrapper) used side-by-side.
