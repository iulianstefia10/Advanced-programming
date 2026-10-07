from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, ForeignKey, Numeric

from app import db

class Artist(db.Model):
    __tablename__ = 'Artist'
    ArtistId: Mapped[int] = mapped_column(Integer, primary_key=True)
    Name: Mapped[str] = mapped_column(String(120))
    albums = relationship('Album', back_populates='artist')

class Album(db.Model):
    __tablename__ = 'Album'
    AlbumId: Mapped[int] = mapped_column(Integer, primary_key=True)
    Title: Mapped[str] = mapped_column(String(160))
    ArtistId: Mapped[int] = mapped_column(Integer, ForeignKey('Artist.ArtistId'))
    artist = relationship('Artist', back_populates='albums')
    tracks = relationship('Track', back_populates='album')

class Genre(db.Model):
    __tablename__ = 'Genre'
    GenreId: Mapped[int] = mapped_column(Integer, primary_key=True)
    Name: Mapped[str] = mapped_column(String(120))
    tracks = relationship('Track', back_populates='genre')

# class InferenceModel(db.Model):
#     __tablename__ = 'InferenceModel'
#     ModelId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(100))
#     FileName: Mapped[str] = mapped_column(String(200))
#     Description: Mapped[str] = mapped_column(String(255), nullable=True)

class Track(db.Model):
    __tablename__ = 'Track'
    TrackId: Mapped[int] = mapped_column(Integer, primary_key=True)
    Name: Mapped[str] = mapped_column(String(200))
    AlbumId: Mapped[int] = mapped_column(Integer, ForeignKey('Album.AlbumId'))
    GenreId: Mapped[int] = mapped_column(Integer, ForeignKey('Genre.GenreId'))
    UnitPrice: Mapped[float] = mapped_column(Numeric)
    album = relationship('Album', back_populates='tracks')
    genre = relationship('Genre', back_populates='tracks')
    invoice_lines = relationship('InvoiceLine', back_populates='track')

class Customer(db.Model):
    __tablename__ = 'Customer'
    CustomerId: Mapped[int] = mapped_column(Integer, primary_key=True)
    FirstName: Mapped[str] = mapped_column(String(40))
    LastName: Mapped[str] = mapped_column(String(20))
    invoices = relationship('Invoice', back_populates='customer')

class Invoice(db.Model):
    __tablename__ = 'Invoice'
    InvoiceId: Mapped[int] = mapped_column(Integer, primary_key=True)
    CustomerId: Mapped[int] = mapped_column(Integer, ForeignKey('Customer.CustomerId'))
    Total: Mapped[float] = mapped_column(Numeric)
    customer = relationship('Customer', back_populates='invoices')
    invoice_lines = relationship('InvoiceLine', back_populates='invoice')

class InvoiceLine(db.Model):
    __tablename__ = 'InvoiceLine'
    InvoiceLineId: Mapped[int] = mapped_column(Integer, primary_key=True)
    InvoiceId: Mapped[int] = mapped_column(Integer, ForeignKey('Invoice.InvoiceId'))
    TrackId: Mapped[int] = mapped_column(Integer, ForeignKey('Track.TrackId'))
    UnitPrice: Mapped[float] = mapped_column(Numeric)
    Quantity: Mapped[int] = mapped_column(Integer)
    invoice = relationship('Invoice', back_populates='invoice_lines')
    track = relationship('Track', back_populates='invoice_lines')