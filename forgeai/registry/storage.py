"""Owns: a thin boto3 client wrapper for S3-compatible model artifact access.

Does not own: on-prem NAS mounts, content encryption policies, or download metering.
"""

from __future__ import annotations

import boto3
from botocore.client import BaseClient

from forgeai.config import Settings


class ObjectStore:
    """Coordinates reads and writes against the configured model registry bucket."""

    def __init__(self, client: BaseClient, bucket: str) -> None:
        """Attach an S3 client and target bucket name."""

        self._client = client
        self._bucket = bucket

    @classmethod
    def from_settings(cls, settings: Settings) -> ObjectStore:
        """Build an object store from validated environment-backed settings."""

        client = boto3.client(
            "s3",
            endpoint_url=str(settings.s3_endpoint_url),
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )
        return cls(client=client, bucket=settings.model_registry_bucket)

    def bucket_name(self) -> str:
        """Return the bucket name used for artifact storage."""

        return self._bucket


__all__ = ["ObjectStore"]
