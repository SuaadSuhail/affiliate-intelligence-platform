"""
PostgreSQL connection and session management.
Uses SQLAlchemy 2.x async-compatible session factory.
"""

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.storage.models import Base
from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://aip_user:aip_secret@localhost:5432/affiliate_intelligence",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # recycle stale connections
    pool_size=10,
    max_overflow=20,
    echo=False,               # set True to log SQL statements
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """Create all tables defined in models.py if they do not exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created / verified.")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a database session and closes it afterwards.

    Usage:
        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context-manager session for use outside FastAPI (scripts, agent tools).

    Usage:
        with db_session() as db:
            affiliates = db.query(Affiliate).all()
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def health_check() -> bool:
    """Return True if the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Health check failed", extra={"error": str(exc)})
        return False
