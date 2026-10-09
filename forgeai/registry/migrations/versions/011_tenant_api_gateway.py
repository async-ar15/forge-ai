"""Add API key lookup and tier columns on ``tenants`` for gateway auth."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "011_tenant_api_gateway"
down_revision = "010_create_training_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable API key material and non-null tier default."""

    op.add_column(
        "tenants",
        sa.Column("api_key_sha256_hex", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "tenants",
        sa.Column("api_key_bcrypt_hash", sa.Text(), nullable=True),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "tenant_tier",
            sa.String(length=32),
            nullable=False,
            server_default="free",
        ),
    )
    op.create_unique_constraint(
        "uq_tenants_api_key_sha256_hex",
        "tenants",
        ["api_key_sha256_hex"],
    )


def downgrade() -> None:
    """Remove gateway auth columns."""

    op.drop_constraint("uq_tenants_api_key_sha256_hex", "tenants", type_="unique")
    op.drop_column("tenants", "tenant_tier")
    op.drop_column("tenants", "api_key_bcrypt_hash")
    op.drop_column("tenants", "api_key_sha256_hex")
