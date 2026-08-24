"""Database engine and session factory for AIMS Layer 5.

Security notes:
  - Database connections MUST use parameterized queries only (enforced by SQLAlchemy ORM).
  - Connection strings MUST NOT be hardcoded. Use environment variables.
  - TODO(security): For production, use mTLS for database connection authentication.
  - TODO(security): For production, use a dedicated DB user with minimal privileges
    (SELECT/INSERT/UPDATE/DELETE only, no DROP/ALTER).

For dev/test: SQLite async via aiosqlite (zero-config, fast).
For production: PostgreSQL async via asyncpg.
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aims.db.tables import Base

# Default to SQLite for development. Override with AIMS_DATABASE_URL env var.
_DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///aims_dev.db"


def get_database_url() -> str:
    """Resolve database URL from environment.

    MUST NOT fall back to a hardcoded production URL.
    The SQLite default is acceptable only for dev/test.
    """
    return os.environ.get("AIMS_DATABASE_URL", _DEFAULT_DATABASE_URL)


def create_engine(database_url: str | None = None):
    """Create an async SQLAlchemy engine."""
    url = database_url or get_database_url()
    # echo=False in production; True only for debugging
    return create_async_engine(url, echo=False)


def create_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the given engine."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db(engine) -> None:
    """Create all tables. For dev/test use only.

    Production should use Alembic migrations.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
