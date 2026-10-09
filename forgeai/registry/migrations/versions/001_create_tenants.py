"""Create the tenants anchor table.

Section 4 references ``tenant_id`` foreign keys from ``routing_decisions`` and
``tenant_policies`` but does not enumerate tenant columns. This migration defines
the minimum surrogate key plus ``created_at`` / ``updated_at`` timestamps required
by the repository migration standards (deviation from Section 4 column list).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001_create_tenants"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``tenants`` with UUID primary key and audit timestamps."""

    op.create_table(
        "tenants",
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
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
        sa.PrimaryKeyConstraint("tenant_id", name="pk_tenants"),
    )


def downgrade() -> None:
    """Drop ``tenants``."""

    op.drop_table("tenants")
