
from flask import Blueprint, render_template, jsonify
from sqlalchemy import func, inspect
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64

from extensions import db
from models.chinook import (
    Artist,
    Album,
    Genre,
    Track,
    Customer,
    Invoice,
    InvoiceLine,
)

chinook_bp = Blueprint(
    "chinook",
    __name__,
    url_prefix="/chinook",
)


@chinook_bp.route("/")
def dashboard():
    return render_template("chinook_dashboard.html")


@chinook_bp.route("/artists")
def artists():
    artist_list = Artist.query.order_by(Artist.Name).all()

    items = [
        {"name": artist.Name, "value": ""}
        for artist in artist_list
        if artist.Name
    ]

    return render_template(
        "data_list.html",
        title="Artists",
        items=items,
        total_count=len(artist_list),
    )


@chinook_bp.route("/albums")
def albums():
    album_list = Album.query.order_by(Album.Title).all()

    items = [
        {"name": album.Title, "value": ""}
        for album in album_list
        if album.Title
    ]

    return render_template(
        "data_list.html",
        title="Albums",
        items=items,
        total_count=len(album_list),
    )


@chinook_bp.route("/artists_with_albums")
def artists_with_albums():
    data = (
        db.session.query(Artist.Name, Album.Title)
        .join(Album, Album.ArtistId == Artist.ArtistId)
        .order_by(Artist.Name, Album.Title)
        .all()
    )

    items = [
        {
            "name": artist_name or "Unknown artist",
            "value": album_title or "Untitled album",
        }
        for artist_name, album_title in data
    ]

    return render_template(
        "data_list.html",
        title="Artists with Albums",
        items=items,
        total_count=len(items),
    )


@chinook_bp.route("/albums_per_artist")
def albums_per_artist():
    data = (
        db.session.query(
            Artist.Name,
            func.count(Album.AlbumId).label("album_count"),
        )
        .outerjoin(Album, Album.ArtistId == Artist.ArtistId)
        .group_by(Artist.ArtistId, Artist.Name)
        .order_by(func.count(Album.AlbumId).desc(), Artist.Name)
        .all()
    )

    items = [
        {
            "name": name or "Unknown artist",
            "value": f"{count} Albums",
        }
        for name, count in data
    ]

    return render_template(
        "data_list.html",
        title="Albums per Artist",
        items=items,
        total_count=len(items),
    )


@chinook_bp.route("/popular_artists")
def popular_artists():
    sales = func.sum(
        InvoiceLine.UnitPrice * InvoiceLine.Quantity
    ).label("sales")

    data = (
        db.session.query(Artist.Name, sales)
        .join(Album, Album.ArtistId == Artist.ArtistId)
        .join(Track, Track.AlbumId == Album.AlbumId)
        .join(InvoiceLine, InvoiceLine.TrackId == Track.TrackId)
        .group_by(Artist.ArtistId, Artist.Name)
        .order_by(sales.desc())
        .limit(10)
        .all()
    )

    items = [
        {
            "name": name or "Unknown artist",
            "value": f"${float(total or 0):.2f}",
        }
        for name, total in data
    ]

    return render_template(
        "data_list.html",
        title="Top 10 Popular Artists (By Sales)",
        items=items,
        total_count=len(items),
    )


@chinook_bp.route("/top_customers")
def top_customers():
    spent = func.sum(Invoice.Total).label("spent")

    data = (
        db.session.query(
            Customer.FirstName,
            Customer.LastName,
            spent,
        )
        .join(Invoice, Invoice.CustomerId == Customer.CustomerId)
        .group_by(
            Customer.CustomerId,
            Customer.FirstName,
            Customer.LastName,
        )
        .order_by(spent.desc())
        .limit(10)
        .all()
    )

    items = [
        {
            "name": f"{first_name or ''} {last_name or ''}".strip(),
            "value": f"${float(total or 0):.2f}",
        }
        for first_name, last_name, total in data
    ]

    return render_template(
        "data_list.html",
        title="Top 10 Spenders",
        items=items,
        total_count=len(items),
    )


@chinook_bp.route("/genres_stats")
def genres_stats():
    data = (
        db.session.query(
            Genre.Name,
            func.count(Track.TrackId).label("track_count"),
        )
        .outerjoin(Track, Track.GenreId == Genre.GenreId)
        .group_by(Genre.GenreId, Genre.Name)
        .order_by(func.count(Track.TrackId).desc())
        .all()
    )

    genres = [name or "Unknown" for name, count in data]
    counts = [count for name, count in data]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#151515")
    ax.set_facecolor("#151515")

    ax.bar(genres, counts, color="#ff304f")
    ax.tick_params(axis="x", colors="white", labelsize=9)
    ax.tick_params(axis="y", colors="white")
    ax.set_ylabel("Number of Tracks", color="white")
    ax.set_title("Tracks per Genre", color="white")

    for spine in ax.spines.values():
        spine.set_color("#555555")

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()

    img = io.BytesIO()
    fig.savefig(
        img,
        format="png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode("utf-8")
    plt.close(fig)

    items = [
        {
            "name": name or "Unknown",
            "value": f"{count} Tracks",
        }
        for name, count in data
    ]

    return render_template(
        "data_list.html",
        title="Genre Statistics",
        items=items,
        total_count=len(items),
        plot_url=plot_url,
    )


@chinook_bp.route("/metadata")
def metadata():
    inspector = inspect(db.engine)
    tables = {}

    for table_name in inspector.get_table_names():
        tables[table_name] = [
            {
                "name": column["name"],
                "type": str(column["type"]),
            }
            for column in inspector.get_columns(table_name)
        ]

    return render_template(
        "metadata.html",
        tables=tables,
    )


@chinook_bp.route("/api/metadata")
def json_metadata():
    inspector = inspect(db.engine)
    tables = {}

    for table_name in inspector.get_table_names():
        tables[table_name] = [
            {
                "name": column["name"],
                "type": str(column["type"]),
            }
            for column in inspector.get_columns(table_name)
        ]

    return jsonify(tables)
