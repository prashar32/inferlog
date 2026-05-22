"""Conversation CRUD — create, list, resume (get), delete."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response

from .db import Database
from .deps import get_db, get_llm
from .llm import LLMRuntime
from .models import (
    ConversationDetail,
    ConversationSummary,
    CreateConversationRequest,
    MessageOut,
)

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


@router.post("", response_model=ConversationDetail, status_code=201)
async def create_conversation(
    body: CreateConversationRequest,
    db: Database = Depends(get_db),
    llm: LLMRuntime = Depends(get_llm),
):
    option = llm.resolve(body.model)
    if option is None:
        raise HTTPException(
            400,
            f"model '{body.model}' is not available; "
            f"see GET /v1/models for the current list",
        )
    conv = await db.create_conversation(
        provider=option.provider,
        model=option.model,
        title=body.title,
        system_prompt=body.system_prompt,
    )
    return ConversationDetail(**conv, messages=[])


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(db: Database = Depends(get_db)):
    return await db.list_conversations()


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: UUID, db: Database = Depends(get_db)):
    conv = await db.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    messages = await db.get_messages(conversation_id)
    return ConversationDetail(
        **conv, messages=[MessageOut(**m) for m in messages]
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: UUID, db: Database = Depends(get_db)):
    if not await db.delete_conversation(conversation_id):
        raise HTTPException(404, "conversation not found")
    return Response(status_code=204)
