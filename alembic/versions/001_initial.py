"""Initial schema

Revision ID: 001
"""
from collections.abc import Sequence

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass  # Tables created via Base.metadata.create_all on startup for R1 dev


def downgrade() -> None:
    pass
