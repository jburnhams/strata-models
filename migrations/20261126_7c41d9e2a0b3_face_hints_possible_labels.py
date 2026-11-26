"""face hints: possible labels an app has for a picture or a face

Revision ID: 7c41d9e2a0b3
Revises: 055ea1c2745c
Create Date: 2026-11-26 16:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c41d9e2a0b3"
down_revision: Union[str, Sequence[str], None] = "055ea1c2745c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "face_hints",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("app", sa.String(length=64), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("app_ref", sa.String(length=256), nullable=False),
        sa.Column("at", sa.Double(), nullable=False),
        sa.Column("face_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=300), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint("status in ('possible','confirmed')", name=op.f("ck_face_hints_status_valid")),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["face_batches.id"], name=op.f("fk_face_hints_batch_id_face_batches"), ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["face_id"], ["faces.id"], name=op.f("fk_face_hints_face_id_faces"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["image_id"], ["face_images.id"], name=op.f("fk_face_hints_image_id_face_images"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_face_hints")),
    )
    with op.batch_alter_table("face_hints", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_face_hints_app"), ["app"], unique=False)
        batch_op.create_index(batch_op.f("ix_face_hints_batch_id"), ["batch_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_face_hints_face_id"), ["face_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_face_hints_image_id"), ["image_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("face_hints", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_face_hints_image_id"))
        batch_op.drop_index(batch_op.f("ix_face_hints_face_id"))
        batch_op.drop_index(batch_op.f("ix_face_hints_batch_id"))
        batch_op.drop_index(batch_op.f("ix_face_hints_app"))
    op.drop_table("face_hints")
