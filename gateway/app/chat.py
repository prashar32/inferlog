"""The chat endpoint: send a message, stream the answer back over SSE.

Cancellation: when the client aborts the request (the Stop button), the
streaming task is cancelled. We catch that, let the SDK record the call as
`cancelled`, and persist whatever partial answer was produced so a resumed
conversation stays honest.
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

import inferlog

from .config import settings
from .db import Database
from .deps import get_db, get_llm
from .llm import ChatMessage, LLMRuntime
from .models import SendMessageRequest

log = logging.getLogger("gateway.chat")
router = APIRouter(tags=["chat"])

# Keep strong references to background persist tasks until they finish,
# otherwise the event loop may garbage-collect them mid-write.
_pending: set[asyncio.Task] = set()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_context(system_prompt: str, history: list[dict]) -> list[ChatMessage]:
    """System prompt + a sliding window of recent turns."""
    messages = [ChatMessage("system", system_prompt)]
    messages += [ChatMessage(m["role"], m["content"]) for m in history]
    return messages


def _usage_dict(usage) -> dict | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


async def _persist_assistant_turn(
    db: Database, conversation_id: UUID, request_id: str, text: str, usage, outcome: str
) -> None:
    """Store the assistant turn. Best-effort — a DB hiccup here must not be
    visible to the user, who already has their answer."""
    try:
        if not text and outcome != "complete":
            # Cancelled/failed before any token — nothing worth a bubble.
            # The inference log still captures that the call happened.
            await db.touch_conversation(conversation_id)
            return
        await db.add_message(
            conversation_id,
            "assistant",
            text,
            status=outcome,
            request_id=request_id,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        await db.touch_conversation(conversation_id)
    except Exception:
        log.exception("could not persist assistant turn for %s", conversation_id)


def _spawn_persist(*args) -> None:
    task = asyncio.create_task(_persist_assistant_turn(*args))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


@router.post("/v1/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: UUID,
    body: SendMessageRequest,
    db: Database = Depends(get_db),
    llm: LLMRuntime = Depends(get_llm),
):
    conv = await db.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(404, "conversation not found")

    content = body.content.strip()
    if not content:
        raise HTTPException(422, "message content is empty")

    # Persist the user turn before generating so it survives a crash mid-stream.
    await db.add_message(conversation_id, "user", content)
    await db.set_title_if_empty(conversation_id, content[:60])

    history = await db.recent_messages(conversation_id, settings.context_window)
    system_prompt = conv["system_prompt"] or settings.default_system_prompt
    context = _build_context(system_prompt, history)
    request_id = str(uuid4())

    async def stream():
        # `with inferlog.context(...)` wraps the async-generator body.
        # contextvars propagate cleanly across `await`/`yield`, so every
        # event emitted by the SDK while we're producing tokens carries
        # the conversation_id tag — both the HTTP-captured (OpenAI /
        # Anthropic) path and the explicit-wrapper (mock) path.
        with inferlog.context(conversation_id=str(conversation_id)):
            collected: list[str] = []
            usage = None
            yield _sse("start", {"request_id": request_id})

            sdk_stream = llm.stream(
                provider=conv["provider"],
                model=conv["model"],
                messages=context,
            )
            try:
                async for chunk in sdk_stream:
                    if chunk.text:
                        collected.append(chunk.text)
                        yield _sse("token", {"text": chunk.text})
                    if chunk.usage:
                        usage = chunk.usage
                # ---- completed normally ----
                await _persist_assistant_turn(
                    db, conversation_id, request_id, "".join(collected), usage, "complete"
                )
                yield _sse("done", {"request_id": request_id, "usage": _usage_dict(usage)})
            except (asyncio.CancelledError, GeneratorExit):
                # User hit Stop / disconnected. Persist the partial answer off
                # the request path — awaiting here during cancellation is unsafe.
                _spawn_persist(
                    db, conversation_id, request_id, "".join(collected), usage, "cancelled"
                )
                raise
            except Exception as exc:
                log.exception("generation failed for conversation %s", conversation_id)
                _spawn_persist(
                    db, conversation_id, request_id, "".join(collected), usage, "error"
                )
                reason = " ".join(str(exc).split())[:240] or "unknown error"
                yield _sse("error", {"message": f"Model request failed — {reason}"})
            finally:
                # Closing the SDK stream is what makes it emit its log line
                # (success/cancelled/error) — so this must always run.
                try:
                    await sdk_stream.aclose()
                except Exception:
                    log.debug("sdk stream close raised", exc_info=True)

    return StreamingResponse(stream(), media_type="text/event-stream")
