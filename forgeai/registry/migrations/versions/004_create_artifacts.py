"""Create the artifacts table for object-storage pointers.

Section 4 states ``model_versions.artifact_id`` references artifacts (S3 path) but
does not tabulate artifact columns. This migration adds ``artifact_id`` plus
``storage_uri`` (non-null textual S3 URI) and audit timestamps — required deviation
to materialize the foreign key target.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "004_create_artifacts"
down_revision = "002_create_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``artifacts``."""

    op.create_table(
        "artifacts",
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("storage_uri", sa.Text(), nullable=False),
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
        sa.PrimaryKeyConstraint("artifact_id", name="pk_artifacts"),
    )


def downgrade() -> None:
    """Drop ``artifacts``."""

    op.drop_table("artifacts")
