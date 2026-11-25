"""SQLAlchemy engine / session handling (PostgreSQL in production, SQLite in tests)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal: sessionmaker | None = None


def configure(url: str) -> None:
    """(Re)bind the engine. Called once at startup, or by tests with a SQLite URL."""
    global _engine, _SessionLocal
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
    _engine = create_engine(url, **kwargs)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def _ensure() -> None:
    if _SessionLocal is None:
        from .config import get_settings
        configure(get_settings().database_url)


def init_db() -> None:
    from .models import Base
    _ensure()
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    _ensure()
    session = _SessionLocal()  # type: ignore[misc]
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping_db() -> bool:
    from sqlalchemy import text
    _ensure()
    with _engine.connect() as conn:  # type: ignore[union-attr]
        conn.execute(text("SELECT 1"))
    return True
