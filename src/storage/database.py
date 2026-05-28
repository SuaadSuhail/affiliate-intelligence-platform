"""
Database connection and session management.

Loads configuration from .env via python-dotenv.
Creates a SQLAlchemy engine with connection pooling and exposes:

  - SessionLocal  : session factory used by all modules
  - get_db()      : FastAPI dependency that yields a session
  - create_all_tables() : idempotent schema creation with clear error
                          message if PostgreSQL is not reachable
"""

import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://aip_user:aip_secret@localhost:5432/affiliate_intelligence",
)

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping recycles connections that have gone stale while idle.
# pool_size / max_overflow keep resource use predictable in local dev.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,  # set True to print generated SQL to stdout
)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # keep objects usable after commit
)


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session and guarantee it is closed when the
    request finishes, even if an exception is raised.

    Usage (FastAPI):
        @app.get("/affiliates")
        def list_affiliates(db: Session = Depends(get_db)):
            return db.query(Affiliate).all()
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Schema creation ───────────────────────────────────────────────────────────

def create_all_tables() -> None:
    """
    Create every table defined in models.py if it does not already exist.

    This function is idempotent — safe to call on every application start.
    Raises OperationalError with a clear message if PostgreSQL is not
    reachable, so the developer knows exactly what to fix.
    """
    # Local import prevents a circular dependency at engine-initialisation
    # time: database.py creates the engine first; models.py can safely import
    # anything it needs without risk of a half-initialised engine.
    from src.storage.models import Base  # noqa: PLC0415

    try:
        Base.metadata.create_all(bind=engine)
        print("[database] Tables created / verified successfully.")
    except OperationalError as exc:
        _reason = getattr(exc, "orig", exc)
        print(
            f"\n[database] ✗ Could not connect to PostgreSQL.\n"
            f"  URL : {DATABASE_URL}\n"
            f"  Why : {_reason}\n\n"
            f"  Fix : start PostgreSQL with\n"
            f"        docker-compose up -d postgres\n"
            f"        and then retry.\n"
        )
        raise