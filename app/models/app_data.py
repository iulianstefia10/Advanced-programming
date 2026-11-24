from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from extensions import db


class StoredModel(db.Model):
    """A Keras digit model (.h5 or .keras) stored in the database."""
    __bind_key__ = "appdata"
    __tablename__ = "stored_model"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str] = mapped_column(String(255), nullable=True)
    filename: Mapped[str] = mapped_column(String(200))   # gives the extension
    size_bytes: Mapped[int] = mapped_column(Integer)
    data: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )