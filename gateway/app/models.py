"""Request/response schemas for the gateway API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    model: str = Field(..., description="Model id from GET /v1/models")
    title: str | None = None
    system_prompt: str | None = None


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)


class ModelOption(BaseModel):
    provider: str
    model: str
    label: str


class ConversationSummary(BaseModel):
    id: UUID
    title: str | None
    provider: str
    model: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message: str | None = None


class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    status: str
    request_id: UUID | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime


class ConversationDetail(BaseModel):
    id: UUID
    title: str | None
    provider: str
    model: str
    system_prompt: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
