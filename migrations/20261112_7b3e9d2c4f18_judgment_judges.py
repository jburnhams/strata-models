"""model judges: eval_judgments rows by a model (judge = its config hash) with its answers (detail)

Revision ID: 7b3e9d2c4f18
Revises: 5c1d2e7f9a30
Create Date: 2026-11-12 05:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7b3e9d2c4f18"
down_revision: Union[str, Sequence[str], None] = "5c1d2e7f9a30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: two nullable columns, so every existing row stays a person's judgment."""
    with op.batch_alter_table("eval_judgments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("judge", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("detail", sa.JSON(), nullable=True))
        batch_op.create_index(batch_op.f("ix_eval_judgments_judge"), ["judge"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("eval_judgments", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_eval_judgments_judge"))
        batch_op.drop_column("detail")
        batch_op.drop_column("judge")
