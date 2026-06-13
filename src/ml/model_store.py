"""
Model Store
===========
Abstracts local-disk and S3-compatible model persistence.

By default (USE_S3=false) models are saved to models/ on local disk.
Set USE_S3=true and provide S3_BUCKET to persist across deployments.

Supports any S3-compatible platform via S3_ENDPOINT_URL:
  AWS S3              — leave S3_ENDPOINT_URL empty
  DigitalOcean Spaces — https://nyc3.digitaloceanspaces.com
  Cloudflare R2       — https://<acct>.r2.cloudflarestorage.com
  Backblaze B2        — https://s3.us-west-002.backblazeb2.com
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib

from src.core.logging_config import get_logger

logger = get_logger(__name__)

USE_S3: bool = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET: str = os.getenv("S3_BUCKET", "")
S3_PREFIX: str = os.getenv("S3_MODEL_PREFIX", "models/")
LOCAL_DIR: Path = Path("models/")


def _get_s3_client():
    """Create an S3-compatible boto3 client.

    For non-AWS platforms set S3_ENDPOINT_URL in .env.
    """
    import boto3

    kwargs = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "region_name": os.getenv("AWS_DEFAULT_REGION", "eu-west-2"),
    }
    endpoint_url = os.getenv("S3_ENDPOINT_URL", "")
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def save_model(model, filename: str) -> None:
    """Save model to local disk, then upload to S3 if USE_S3=true."""
    local_path = LOCAL_DIR / filename
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, local_path)
    logger.info("Model saved locally", extra={"path": str(local_path)})

    if USE_S3 and S3_BUCKET:
        s3 = _get_s3_client()
        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        logger.info(
            "Model uploaded to S3",
            extra={"bucket": S3_BUCKET, "key": s3_key},
        )


def load_model(filename: str) -> Optional[object]:
    """Load model from local disk, falling back to S3 if USE_S3=true."""
    local_path = LOCAL_DIR / filename

    if USE_S3 and S3_BUCKET and not local_path.exists():
        logger.info("Downloading model from S3", extra={"filename": filename})
        s3 = _get_s3_client()
        s3_key = f"{S3_PREFIX}{filename}"
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(S3_BUCKET, s3_key, str(local_path))
        except Exception as exc:
            logger.warning(
                "S3 download failed",
                extra={"filename": filename, "error": str(exc)},
            )
            return None

    if local_path.exists():
        return joblib.load(local_path)

    logger.warning("Model not found", extra={"filename": filename})
    return None


def model_exists(filename: str) -> bool:
    """Return True if the model is available locally or in S3."""
    local_path = LOCAL_DIR / filename
    if local_path.exists():
        return True
    if USE_S3 and S3_BUCKET:
        try:
            s3 = _get_s3_client()
            s3.head_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}{filename}")
            return True
        except Exception:
            return False
    return False