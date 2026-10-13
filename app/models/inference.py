from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String

from app import db


class InferenceModel(db.Model):
    __tablename__ = 'InferenceModel'

    ModelId: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    Name: Mapped[str] = mapped_column(
        String(100)
    )

    FileName: Mapped[str] = mapped_column(
        String(200)
    )

    Description: Mapped[str] = mapped_column(
        String(255),
        nullable=True
    )