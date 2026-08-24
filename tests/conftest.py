"""Shared test fixtures for AIMS test suite."""

from __future__ import annotations

import uuid

import pytest

from aims.db.engine import create_engine, create_session_factory, init_db
from aims.db.models import CallerContext


@pytest.fixture
async def engine():
    """In-memory SQLite for test isolation. Each test gets a fresh DB."""
    eng = create_engine("sqlite+aiosqlite:///:memory:")
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session(engine):
    """Async session, rolled back after each test."""
    factory = create_session_factory(engine)
    async with factory() as sess:
        yield sess
        await sess.rollback()


@pytest.fixture
def caller_a() -> CallerContext:
    """Test caller A — represents user A."""
    return CallerContext(user_id=uuid.uuid4())


@pytest.fixture
def caller_b() -> CallerContext:
    """Test caller B — represents user B (different from A)."""
    return CallerContext(user_id=uuid.uuid4())
