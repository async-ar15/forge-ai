"""Create exploitative-only routing decisions view for offline evaluation."""

from __future__ import annotations

from alembic import op

revision = "012_create_eval_view"
down_revision = "011_tenant_api_gateway"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create routing_decisions_exploitative view with mandatory filter."""

    op.execute(
        """
        CREATE VIEW routing_decisions_exploitative AS
        SELECT *
        FROM routing_decisions
        WHERE exploration_flag = FALSE
        """
    )


def downgrade() -> None:
    """Drop routing_decisions_exploitative view."""

    op.execute("DROP VIEW IF EXISTS routing_decisions_exploitative")
