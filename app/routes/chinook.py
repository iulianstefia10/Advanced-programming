from flask import Blueprint, render_template
from sqlalchemy import func
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
    InvoiceLine
)

chinook_bp = Blueprint(
    "chinook",
    __name__,
    url_prefix="/chinook"
)


@chinook_bp.route('/artists')
def artists():
    artist_list = Artist.query.all()
    return f"<h1>List of Artists ({len(artist_list)})</h1>" + \
           "<br>".join([a.Name for a in artist_list if a.Name])


@chinook_bp.route('/albums')
def albums():
    album_list = Album.query.all()
    return f"<h1>List of Albums ({len(album_list)})</h1>" + \
           "<br>".join([a.Title for a in album_list])


@chinook_bp.route('/artists_with_albums')
def artists_with_albums():
    artist_list = Artist.query.join(Album).all()
    return render_template(
        'artists_with_albums.html',
        artists=artist_list
    )


@chinook_bp.route('/')
def dashboard():
    return render_template('dashboard.html')


@chinook_bp.route('/albums_per_artist')
def albums_per_artist():
    data = db.session.query(
        Artist.Name,
        func.count(Album.AlbumId).label('album_count')
    ).outerjoin(Album) \
     .group_by(Artist.ArtistId) \
     .order_by(db.desc('album_count')).all()

    items = [
        {"name": name, "value": f"{count} Albums"}
        for name, count in data
    ]

    return render_template(
        'data_list.html',
        title="Albums per Artist",
        items=items
    )


@chinook_bp.route('/popular_artists')
def popular_artists():
    data = db.session.query(
        Artist.Name,
        func.sum(
            InvoiceLine.UnitPrice * InvoiceLine.Quantity
        ).label('sales')
    ).join(Album) \
     .join(Track) \
     .join(InvoiceLine) \
     .group_by(Artist.Name) \
     .order_by(db.desc('sales')) \
     .limit(10).all()

    items = [
        {"name": name, "value": f"${sales:.2f}"}
        for name, sales in data
    ]

    return render_template(
        'data_list.html',
        title="Top 10 Popular Artists (By Sales)",
        items=items
    )


@chinook_bp.route('/top_customers')
def top_customers():
    data = db.session.query(
        Customer.FirstName,
        Customer.LastName,
        func.sum(Invoice.Total).label('spent')
    ).join(Invoice) \
     .group_by(Customer.CustomerId) \
     .order_by(db.desc('spent')) \
     .limit(10).all()

    items = [
        {
            "name": f"{fname} {lname}",
            "value": f"${spent:.2f}"
        }
        for fname, lname, spent in data
    ]

    return render_template(
        'data_list.html',
        title="Top 10 Spenders",
        items=items
    )


@chinook_bp.route('/genres_stats')
def genres_stats():
    data = db.session.query(
        Genre.Name,
        func.count(Track.TrackId).label('track_count')
    ).join(Track) \
     .group_by(Genre.GenreId) \
     .order_by(db.desc('track_count')).all()

    genres = [d[0] for d in data]
    counts = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(12, 6))

    fig.patch.set_facecolor('#242424')
    ax.set_facecolor('#242424')

    bars = ax.bar(genres, counts, color='#dc3545')

    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Number of Tracks')
    plt.title('Tracks per Genre')

    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')

    plt.tight_layout()

    img = io.BytesIO()
    fig.savefig(
        img,
        format='png',
        facecolor=fig.get_facecolor(),
        edgecolor='none'
    )

    img.seek(0)

    plot_url = base64.b64encode(
        img.getvalue()
    ).decode('utf8')

    plt.close(fig)

    items = [
        {"name": g, "value": f"{c} Tracks"}
        for g, c in data
    ]

    return render_template(
        'data_list.html',
        title="Genre Statistics",
        items=items,
        plot_url=plot_url
    )