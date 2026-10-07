"""Cloudflare R2 access (S3-compatible): manifest fetch and model download into the disk cache."""

import json
import logging
import os
import shutil
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import ValidationError

from config import Settings
from schemas import Manifest

logger = logging.getLogger("nomaapp.r2")

MANIFEST_KEY = "manifest.json"


class R2Error(Exception):
    """R2 is unconfigured, unreachable, or the object could not be read."""


class R2Client:
    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.r2_bucket_name
        self._client = None
        if settings.r2_configured:
            self._client = boto3.client(
                "s3",
                endpoint_url=settings.r2_endpoint_url,
                aws_access_key_id=settings.r2_access_key.get_secret_value(),
                aws_secret_access_key=settings.r2_secret_key.get_secret_value(),
                region_name="auto",
                config=Config(
                    connect_timeout=settings.r2_connect_timeout_s,
                    read_timeout=settings.r2_read_timeout_s,
                    retries={"max_attempts": settings.r2_max_attempts, "mode": "standard"},
                ),
            )
        else:
            logger.warning("R2 credentials not set; only the local disk cache can be used")

    @property
    def configured(self) -> bool:
        return self._client is not None

    def fetch_manifest(self) -> Manifest:
        if self._client is None:
            raise R2Error("R2 is not configured")
        try:
            body = self._client.get_object(Bucket=self._bucket, Key=MANIFEST_KEY)["Body"].read()
            return Manifest.model_validate(json.loads(body))
        except (BotoCoreError, ClientError) as exc:
            raise R2Error(f"manifest fetch failed: {exc}") from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            raise R2Error(f"remote manifest is invalid: {exc}") from exc

    def download(self, key: str, dest: Path) -> int:
        """Download an object to dest atomically (temp file + rename). Returns bytes written."""
        if self._client is None:
            raise R2Error("R2 is not configured")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            self._client.download_file(self._bucket, key, str(tmp))
            os.replace(tmp, dest)
        except (BotoCoreError, ClientError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise R2Error(f"download of {key} failed: {exc}") from exc
        size = dest.stat().st_size
        logger.info("model downloaded from R2", extra={"key": key, "bytes": size})
        return size


def local_path_for(cache_dir: Path, key: str) -> Path:
    """Disk-cache path for a bucket key, mirroring the bucket layout.

    Rejects keys that would escape the cache directory (e.g. "../x").
    """
    root = cache_dir.resolve()
    path = (root / key).resolve()
    if root not in path.parents:
        raise ValueError(f"model key escapes cache dir: {key!r}")
    return path


def _version_tuple(version: str) -> tuple:
    parts = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _read_local_manifest(path: Path) -> Manifest | None:
    if not path.exists():
        return None
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        logger.warning("local manifest unreadable", extra={"error": str(exc)})
        return None


def _write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(r2: R2Client, cache_dir: Path) -> Manifest:
    """Fetch the manifest from R2; fall back to the disk copy if R2 is unreachable.

    Raises RuntimeError if neither is available - the service cannot run without one.
    """
    local_path = cache_dir / MANIFEST_KEY
    local = _read_local_manifest(local_path)

    try:
        remote = r2.fetch_manifest()
    except R2Error as exc:
        if local is None:
            raise RuntimeError(
                "No manifest available: R2 unreachable and no local model_cache/manifest.json"
            ) from exc
        logger.warning(
            "R2 unreachable, using cached manifest",
            extra={"error": str(exc), "manifest_version": local.version},
        )
        return local

    if local is not None and _version_tuple(remote.version) > _version_tuple(local.version):
        logger.info(f"[Manifest] Upgraded from {local.version} to {remote.version}")
    elif local is not None and _version_tuple(remote.version) < _version_tuple(local.version):
        logger.warning(f"[Manifest] Remote version {remote.version} is older than cached {local.version}")
    _write_manifest(local_path, remote)
    logger.info("manifest loaded from R2", extra={"manifest_version": remote.version})
    return remote


def log_disk_status(cache_dir: Path) -> None:
    usage = shutil.disk_usage(cache_dir)
    logger.info(
        "disk cache ready",
        extra={
            "cache_dir": str(cache_dir.resolve()),
            "disk_free_gb": round(usage.free / 1024**3, 2),
            "disk_total_gb": round(usage.total / 1024**3, 2),
        },
    )
