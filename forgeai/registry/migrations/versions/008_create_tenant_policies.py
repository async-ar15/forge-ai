"""Create the tenant_policies table.

Section 4 defines coefficient columns and ``updated_at`` only. This migration
adds ``created_at`` with timezone default ``now()`` to satisfy repository-wide
timestamp requirements (explicit deviation).

Primary key ``tenant_id`` references ``tenants`` with ON DELETE CASCADE so policy
rows are removed when a tenant row is removed (tenant table is administrative;
routing_decisions remain protected by RESTRICT on the tenant FK in migration
009).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "008_create_tenant_policies"
down_revision = "007_create_deployments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``tenant_policies``."""

    op.create_table(
        "tenant_policies",
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("coeff_latency", sa.Float(), nullable=False),
        sa.Column("coeff_cost", sa.Float(), nullable=False),
        sa.Column("coeff_slo_violation", sa.Float(), nullable=False),
        sa.Column("coeff_fallback", sa.Float(), nullable=False),
        sa.Column("coeff_retry", sa.Float(), nullable=False),
        sa.Column("coeff_cache_hit", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            name="fk_tenant_policies_tenant_id_tenants",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("tenant_id", name="pk_tenant_policies"),
    )


def downgrade() -> None:
    """Drop ``tenant_policies``."""

    op.drop_table("tenant_policies")
