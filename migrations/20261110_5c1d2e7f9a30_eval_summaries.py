"""eval summaries: each experiment's analysis, worked out in the background and read by the page

Revision ID: 5c1d2e7f9a30
Revises: 0a13067bc436
Create Date: 2026-11-10 07:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5c1d2e7f9a30"
down_revision: Union[str, Sequence[str], None] = "0a13067bc436"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "eval_summaries",
        sa.Column("experiment", sa.String(length=128), nullable=False),
        sa.Column("signature", sa.String(length=64), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("computed_at", sa.Double(), nullable=False),
        sa.Column("seconds", sa.Double(), nullable=False),
        sa.PrimaryKeyConstraint("experiment", name=op.f("pk_eval_summaries")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("eval_summaries")
