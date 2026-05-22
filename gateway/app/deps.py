"""FastAPI dependencies — pull shared singletons off app.state."""

from fastapi import Request

from .db import Database
from .llm import LLMRuntime


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_llm(request: Request) -> LLMRuntime:
    return request.app.state.llm
