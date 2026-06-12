"""API key authentication dependency for FastAPI."""
import os

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from src.core.logging_config import get_logger

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(x_api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency — enforces API key auth unless APP_ENV=development."""
    if os.getenv("APP_ENV", "development") == "development":
        return "dev-bypass"

    secret = os.getenv("API_SECRET_KEY", "")
    if not secret:
        logger.error("API_SECRET_KEY not set — refusing protected request")
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: API key not configured",
        )

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    if x_api_key != secret:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key