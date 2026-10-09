"""Update exploitative eval view to exclude internal tenant tier."""

from __future__ import annotations

from alembic import op

revision = "013_update_eval_view"
down_revision = "012_create_eval_view"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Exclude internal-tier system traffic from offline eval view."""

    op.execute("DROP VIEW IF EXISTS routing_decisions_exploitative")
    op.execute(
        """
        CREATE VIEW routing_decisions_exploitative AS
        SELECT *
        FROM routing_decisions
        WHERE exploration_flag = FALSE
          AND tenant_tier != 'internal'
        """
    )


def downgrade() -> None:
    """Restore prior eval view without internal-tier exclusion."""

    op.execute("DROP VIEW IF EXISTS routing_decisions_exploitative")
    op.execute(
        """
        CREATE VIEW routing_decisions_exploitative AS
        SELECT *
        FROM routing_decisions
        WHERE exploration_flag = FALSE
        """
    )
