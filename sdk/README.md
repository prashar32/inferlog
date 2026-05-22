# inferlog

A small wrapper that turns LLM provider calls into structured inference
logs. It is consumed by the gateway service in this repo, but it has no
dependency on the rest of the system — the only required package is `httpx`.

## What it does

- Wraps a provider call (`complete` or `stream`) and captures: model,
  provider, latency, time-to-first-token, token usage, status
  (success / error / cancelled), timestamps, conversation id, and
  truncated input/output previews.
- Hands each event to a background `LogDispatcher` that batches, retries
  with backoff, and ships them to an ingestion endpoint.
- Never blocks or breaks the caller: `submit()` is non-blocking and sheds
  load if the queue fills up.

## Providers

`openai`, `anthropic`, and `mock`. The vendor SDKs are optional extras
(`pip install inferlog[openai]`); `mock` needs nothing and is what makes
the system runnable and testable offline.

## Example

```python
from inferlog import LoggedLLMClient, LogDispatcher, HttpSink
from inferlog.providers import MockProvider

dispatcher = LogDispatcher(HttpSink("http://ingestion:8080/v1/ingest"))
dispatcher.start()

client = LoggedLLMClient(
    service="chat-gateway",
    dispatcher=dispatcher,
    providers={"mock": MockProvider()},
)

async for chunk in client.stream(
    provider="mock", model="mock-1",
    messages=[ChatMessage("user", "hello")], conversation_id="abc",
):
    print(chunk.text, end="")
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
