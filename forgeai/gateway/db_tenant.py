"""Postgres lookup for tenant rows keyed by API key hash."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forgeai.registry.models.tenant import Tenant


async def load_tenant_by_api_key_sha(
    session: AsyncSession,
    api_key_sha256_hex: str,
) -> Tenant | None:
    """Return the tenant row for a hex SHA-256 of the raw API key, if any."""

    stmt = select(Tenant).where(Tenant.api_key_sha256_hex == api_key_sha256_hex)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


def tenant_id_uuid(row: Tenant) -> uuid.UUID:
    """Narrow tenant_id to UUID."""

    return row.tenant_id


__all__ = ["load_tenant_by_api_key_sha", "tenant_id_uuid"]
