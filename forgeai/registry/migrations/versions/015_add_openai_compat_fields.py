"""Add OpenAI compatibility hint columns to routing_decisions."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "015_add_openai_compat_fields"
down_revision = "014_create_training_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "routing_decisions",
        sa.Column("requested_model_hint", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "routing_decisions",
        sa.Column("endpoint", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("routing_decisions", "endpoint")
    op.drop_column("routing_decisions", "requested_model_hint")
