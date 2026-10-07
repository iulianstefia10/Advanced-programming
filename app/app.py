from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from influxdb_flask import InfluxDB
import os

from PIL import Image
import numpy as np
from keras.models import load_model

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, ForeignKey, Numeric, func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    basedir, "database/chinook.db"
)
db = SQLAlchemy(app)

# class Artist(db.Model):
#     __tablename__ = 'Artist'
#     ArtistId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(120))
#     albums = relationship('Album', back_populates='artist')

# class Album(db.Model):
#     __tablename__ = 'Album'
#     AlbumId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Title: Mapped[str] = mapped_column(String(160))
#     ArtistId: Mapped[int] = mapped_column(Integer, ForeignKey('Artist.ArtistId'))
#     artist = relationship('Artist', back_populates='albums')
#     tracks = relationship('Track', back_populates='album')

# class Genre(db.Model):
#     __tablename__ = 'Genre'
#     GenreId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(120))
#     tracks = relationship('Track', back_populates='genre')

# class InferenceModel(db.Model):
#     __tablename__ = 'InferenceModel'
#     ModelId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(100))
#     FileName: Mapped[str] = mapped_column(String(200))
#     Description: Mapped[str] = mapped_column(String(255), nullable=True)

# class Track(db.Model):
#     __tablename__ = 'Track'
#     TrackId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(200))
#     AlbumId: Mapped[int] = mapped_column(Integer, ForeignKey('Album.AlbumId'))
#     GenreId: Mapped[int] = mapped_column(Integer, ForeignKey('Genre.GenreId'))
#     UnitPrice: Mapped[float] = mapped_column(Numeric)
#     album = relationship('Album', back_populates='tracks')
#     genre = relationship('Genre', back_populates='tracks')
#     invoice_lines = relationship('InvoiceLine', back_populates='track')

# class Customer(db.Model):
#     __tablename__ = 'Customer'
#     CustomerId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     FirstName: Mapped[str] = mapped_column(String(40))
#     LastName: Mapped[str] = mapped_column(String(20))
#     invoices = relationship('Invoice', back_populates='customer')

# class Invoice(db.Model):
#     __tablename__ = 'Invoice'
#     InvoiceId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     CustomerId: Mapped[int] = mapped_column(Integer, ForeignKey('Customer.CustomerId'))
#     Total: Mapped[float] = mapped_column(Numeric)
#     customer = relationship('Customer', back_populates='invoices')
#     invoice_lines = relationship('InvoiceLine', back_populates='invoice')

# class InvoiceLine(db.Model):
#     __tablename__ = 'InvoiceLine'
#     InvoiceLineId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     InvoiceId: Mapped[int] = mapped_column(Integer, ForeignKey('Invoice.InvoiceId'))
#     TrackId: Mapped[int] = mapped_column(Integer, ForeignKey('Track.TrackId'))
#     UnitPrice: Mapped[float] = mapped_column(Numeric)
#     Quantity: Mapped[int] = mapped_column(Integer)
#     invoice = relationship('Invoice', back_populates='invoice_lines')
#     track = relationship('Track', back_populates='invoice_lines')


app.config["INFLUXDB_V2_URL"] = "http://192.168.168.88:8086"
app.config["INFLUXDB_V2_ORG"] = "iut"
app.config["INFLUXDB_V2_TOKEN"] = "YOUR_TOKEN"

influxdb = InfluxDB(app)

with app.app_context():
    db.create_all()
    
    if not InferenceModel.query.first():
        mnist = InferenceModel(Name="Model MNIST", FileName="MNIST_keras_CNN.h5", Description="Recunoastere cifre")
        db.session.add(mnist)
        db.session.commit()

    print("Tables in the database:")
    for table_name in db.metadata.tables:
        print(f"- {table_name}")

    for table_name, table in db.metadata.tables.items():
        print(f"\nTable: {table_name}")
        print("Columns:")
        for column in table.columns:
            print(f"  - {column.name} ({column.type})")


def get_metadata():
    with app.app_context():
        tables_data = {}
        for table_name, table in db.metadata.tables.items():
            tables_data[table_name] = [
                {"name": col.name, "type": str(col.type)} for col in table.columns
            ]
        return tables_data


@app.route('/')
def display_metadata():
    data = get_metadata()
    return render_template('metadata.html', tables=data)

@app.route('/api/metadata')
def json_metadata():
    data = get_metadata()
    return jsonify(data)


@app.route('/artists')
def artists():
    artist_list = Artist.query.all()
    return f"<h1>List of Artists ({len(artist_list)})</h1>" + "<br>".join([a.Name for a in artist_list if a.Name])

@app.route('/albums')
def albums():
    album_list = Album.query.all()
    return f"<h1>List of Albums ({len(album_list)})</h1>" + "<br>".join([a.Title for a in album_list])

@app.route('/artists_with_albums')
def artists_with_albums():
    artist_list = Artist.query.join(Album).all()
    return render_template('artists_with_albums.html', artists=artist_list)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/albums_per_artist')
def albums_per_artist():
    data = db.session.query(
        Artist.Name, 
        func.count(Album.AlbumId).label('album_count')
    ).outerjoin(Album).group_by(Artist.ArtistId).order_by(db.desc('album_count')).all()
    
    items = [{"name": name, "value": f"{count} Albums"} for name, count in data]
    return render_template('data_list.html', title="Albums per Artist", items=items)

@app.route('/popular_artists')
def popular_artists():
    data = db.session.query(
        Artist.Name, 
        func.sum(InvoiceLine.UnitPrice * InvoiceLine.Quantity).label('sales')
    ).join(Album).join(Track).join(InvoiceLine).group_by(Artist.Name).order_by(db.desc('sales')).limit(10).all()
    
    items = [{"name": name, "value": f"${sales:.2f}"} for name, sales in data]
    return render_template('data_list.html', title="Top 10 Popular Artists (By Sales)", items=items)

@app.route('/top_customers')
def top_customers():
    data = db.session.query(
        Customer.FirstName, Customer.LastName, func.sum(Invoice.Total).label('spent')
    ).join(Invoice).group_by(Customer.CustomerId).order_by(db.desc('spent')).limit(10).all()
    
    items = [{"name": f"{fname} {lname}", "value": f"${spent:.2f}"} for fname, lname, spent in data]
    return render_template('data_list.html', title="Top 10 Spenders", items=items)

@app.route('/genres_stats')
def genres_stats():
    data = db.session.query(
        Genre.Name, 
        func.count(Track.TrackId).label('track_count')
    ).join(Track).group_by(Genre.GenreId).order_by(db.desc('track_count')).all()
    
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
    fig.savefig(img, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig) 
    
    items = [{"name": g, "value": f"{c} Tracks"} for g, c in data]
    return render_template('data_list.html', title="Genre Statistics", items=items, plot_url=plot_url)


@app.route("/ping")
def ping():
    response = influxdb.connection.ping()
    return {"connected": response}

@app.route("/buckets")
def buckets():
    buckets_api = influxdb.connection.buckets_api()
    buckets = buckets_api.find_buckets().buckets
    return {"buckets": [bucket.name for bucket in buckets]}

@app.route("/weather")
def weather():
    query_api = influxdb.connection.query_api()
    org = app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    from(bucket: "weather")
      |> range(start: -1w)
      |> filter(fn: (r) => r._measurement == "weather_current")
      |> filter(fn: (r) => r._field == "temperature_2m")
      |> filter(fn: (r) => r.location == "annecy")
    '''

    tables = query_api.query(
        org=org,
        query=flux_query
    )

    data = [
        record.values["_value"]
        for table in tables
        for record in table.records
    ]

    time = [
        record.values["_time"]
        for table in tables
        for record in table.records
    ]

    return {
        "data": data,
        "time": [str(t) for t in time]
    }


@app.route("/measurements")
def measurements():
    query_api = influxdb.connection.query_api()
    org = app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    import "influxdata/influxdb/schema"

    schema.measurements(bucket: "weather")
    '''

    tables = query_api.query(
        org=org,
        query=flux_query
    )

    meas_list = [
        record.values["_value"]
        for table in tables
        for record in table.records
    ]

    return {"measurements": meas_list}

@app.route("/fields")
def fields():
    query_api = influxdb.connection.query_api()
    org = app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    import "influxdata/influxdb/schema"

    schema.fieldKeys(bucket: "weather")
    '''

    tables = query_api.query(
        org=org,
        query=flux_query
    )

    field_list = [
        record.values["_value"]
        for table in tables
        for record in table.records
    ]

    return {"fields": field_list}

def format_dataset(data, x):
    X = []
    y = []

    for i in range(len(data) - 2 * x + 1):
        X.append(data[i:i + x])
        y.append(data[i + x:i + 2 * x])

    return X, y

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5047,
        debug=True
    )






# from flask import Flask, render_template, jsonify, request
# from flask_sqlalchemy import SQLAlchemy
# from influxdb_flask import InfluxDB
# import os

# from PIL import Image
# import numpy as np
# from keras.models import load_model

# from sqlalchemy.orm import Mapped, mapped_column, relationship
# from sqlalchemy import Integer, String, ForeignKey, Numeric, func

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import io
# import base64

# basedir = os.path.abspath(os.path.dirname(__file__))

# app = Flask(__name__)

# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
#     basedir, "database/chinook.db"
# )

# db = SQLAlchemy(app)

# class Artist(db.Model):
#     __tablename__ = 'Artist'

#     ArtistId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(120))

#     albums = relationship('Album', back_populates='artist')

# class Album(db.Model):
#     __tablename__ = 'Album'

#     AlbumId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Title: Mapped[str] = mapped_column(String(160))
#     ArtistId: Mapped[int] = mapped_column(
#         Integer,
#         ForeignKey('Artist.ArtistId')
#     )

#     artist = relationship('Artist', back_populates='albums')
#     tracks = relationship('Track', back_populates='album')

# class Genre(db.Model):
#     __tablename__ = 'Genre'

#     GenreId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(120))

#     tracks = relationship('Track', back_populates='genre')

# class InferenceModel(db.Model):
#     __tablename__ = 'InferenceModel'

#     ModelId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(100))
#     FileName: Mapped[str] = mapped_column(String(200))
#     Description: Mapped[str] = mapped_column(
#         String(255),
#         nullable=True
#     )

# class Track(db.Model):
#     __tablename__ = 'Track'

#     TrackId: Mapped[int] = mapped_column(Integer, primary_key=True)
#     Name: Mapped[str] = mapped_column(String(200))
#     AlbumId: Mapped[int] = mapped_column(
#         Integer,
#         ForeignKey('Album.AlbumId')
#     )
#     GenreId: Mapped[int] = mapped_column(
#         Integer,
#         ForeignKey('Genre.GenreId')
#     )
#     UnitPrice: Mapped[float] = mapped_column(Numeric)

#     album = relationship('Album', back_populates='tracks')
#     genre = relationship('Genre', back_populates='tracks')
#     invoice_lines = relationship(
#         'InvoiceLine',
#         back_populates='track'
#     )

# class Customer(db.Model):
#     __tablename__ = 'Customer'

#     CustomerId: Mapped[int] = mapped_column(
#         Integer,
#         primary_key=True
#     )
#     FirstName: Mapped[str] = mapped_column(String(40))
#     LastName: Mapped[str] = mapped_column(String(20))

#     invoices = relationship(
#         'Invoice',
#         back_populates='customer'
#     )

# class Invoice(db.Model):
#     __tablename__ = 'Invoice'

#     InvoiceId: Mapped[int] = mapped_column(
#         Integer,
#         primary_key=True
#     )
#     CustomerId: Mapped[int] = mapped_column(
#         Integer,
#         ForeignKey('Customer.CustomerId')
#     )
#     Total: Mapped[float] = mapped_column(Numeric)

#     customer = relationship(
#         'Customer',
#         back_populates='invoices'
#     )

#     invoice_lines = relationship(
#         'InvoiceLine',
#         back_populates='invoice'
#     )

# class InvoiceLine(db.Model):
#     __tablename__ = 'InvoiceLine'

#     InvoiceLineId: Mapped[int] = mapped_column(
#         Integer,
#         primary_key=True
#     )
#     InvoiceId: Mapped[int] = mapped_column(
#         Integer,
#         ForeignKey('Invoice.InvoiceId')
#     )
#     TrackId: Mapped[int] = mapped_column(
#         Integer,
#         ForeignKey('Track.TrackId')
#     )
#     UnitPrice: Mapped[float] = mapped_column(Numeric)
#     Quantity: Mapped[int] = mapped_column(Integer)

#     invoice = relationship(
#         'Invoice',
#         back_populates='invoice_lines'
#     )

#     track = relationship(
#         'Track',
#         back_populates='invoice_lines'
#     )

# app.config["INFLUXDB_V2_URL"] = "http://192.168.168.88:8086"
# app.config["INFLUXDB_V2_ORG"] = "iut"
# app.config["INFLUXDB_V2_TOKEN"] = os.environ.get(
#     "INFLUXDB_TOKEN"
# )

# influxdb = InfluxDB(app)

# with app.app_context():

#     db.create_all()

#     if not InferenceModel.query.first():

#         mnist = InferenceModel(
#             Name="Model MNIST",
#             FileName="MNIST_keras_CNN.h5",
#             Description="Recunoastere cifre"
#         )

#         db.session.add(mnist)
#         db.session.commit()

# def get_metadata():

#     with app.app_context():

#         tables_data = {}

#         for table_name, table in db.metadata.tables.items():

#             tables_data[table_name] = [
#                 {
#                     "name": col.name,
#                     "type": str(col.type)
#                 }
#                 for col in table.columns
#             ]

#         return tables_data

# @app.route('/')
# def display_metadata():

#     data = get_metadata()

#     return render_template(
#         'metadata.html',
#         tables=data
#     )

# @app.route('/api/metadata')
# def json_metadata():

#     return jsonify(get_metadata())

# @app.route('/artists')
# def artists():

#     artist_list = Artist.query.all()

#     return (
#         f"<h1>List of Artists ({len(artist_list)})</h1>"
#         +
#         "<br>".join(
#             [
#                 a.Name
#                 for a in artist_list
#                 if a.Name
#             ]
#         )
#     )

# @app.route('/albums')
# def albums():

#     album_list = Album.query.all()

#     return (
#         f"<h1>List of Albums ({len(album_list)})</h1>"
#         +
#         "<br>".join(
#             [
#                 a.Title
#                 for a in album_list
#             ]
#         )
#     )

# @app.route('/artists_with_albums')
# def artists_with_albums():

#     artist_list = Artist.query.join(Album).all()

#     return render_template(
#         'artists_with_albums.html',
#         artists=artist_list
#     )

# @app.route('/dashboard')
# def dashboard():

#     return render_template('dashboard.html')

# @app.route('/albums_per_artist')
# def albums_per_artist():

#     data = (
#         db.session.query(
#             Artist.Name,
#             func.count(Album.AlbumId).label('album_count')
#         )
#         .outerjoin(Album)
#         .group_by(Artist.ArtistId)
#         .order_by(db.desc('album_count'))
#         .all()
#     )

#     items = [
#         {
#             "name": name,
#             "value": f"{count} Albums"
#         }
#         for name, count in data
#     ]

#     return render_template(
#         'data_list.html',
#         title="Albums per Artist",
#         items=items
#     )

# @app.route('/popular_artists')
# def popular_artists():

#     data = (
#         db.session.query(
#             Artist.Name,
#             func.sum(
#                 InvoiceLine.UnitPrice *
#                 InvoiceLine.Quantity
#             ).label('sales')
#         )
#         .join(Album)
#         .join(Track)
#         .join(InvoiceLine)
#         .group_by(Artist.Name)
#         .order_by(db.desc('sales'))
#         .limit(10)
#         .all()
#     )

#     items = [
#         {
#             "name": name,
#             "value": f"${sales:.2f}"
#         }
#         for name, sales in data
#     ]

#     return render_template(
#         'data_list.html',
#         title="Top 10 Popular Artists (By Sales)",
#         items=items
#     )

# @app.route('/top_customers')
# def top_customers():

#     data = (
#         db.session.query(
#             Customer.FirstName,
#             Customer.LastName,
#             func.sum(Invoice.Total).label('spent')
#         )
#         .join(Invoice)
#         .group_by(Customer.CustomerId)
#         .order_by(db.desc('spent'))
#         .limit(10)
#         .all()
#     )

#     items = [
#         {
#             "name": f"{fname} {lname}",
#             "value": f"${spent:.2f}"
#         }
#         for fname, lname, spent in data
#     ]

#     return render_template(
#         'data_list.html',
#         title="Top 10 Spenders",
#         items=items
#     )

# @app.route('/genres_stats')
# def genres_stats():

#     data = (
#         db.session.query(
#             Genre.Name,
#             func.count(Track.TrackId).label('track_count')
#         )
#         .join(Track)
#         .group_by(Genre.GenreId)
#         .order_by(db.desc('track_count'))
#         .all()
#     )

#     genres = [d[0] for d in data]
#     counts = [d[1] for d in data]

#     fig, ax = plt.subplots(figsize=(12, 6))

#     fig.patch.set_facecolor('#242424')
#     ax.set_facecolor('#242424')

#     ax.bar(
#         genres,
#         counts,
#         color='#dc3545'
#     )

#     ax.tick_params(colors='white')

#     ax.xaxis.label.set_color('white')
#     ax.yaxis.label.set_color('white')
#     ax.title.set_color('white')

#     plt.xticks(
#         rotation=90,
#         fontsize=10
#     )

#     plt.yticks(fontsize=10)

#     plt.ylabel('Number of Tracks')
#     plt.title('Tracks per Genre')

#     for spine in ax.spines.values():
#         spine.set_edgecolor('#555555')

#     plt.tight_layout()

#     img = io.BytesIO()

#     fig.savefig(
#         img,
#         format='png',
#         facecolor=fig.get_facecolor(),
#         edgecolor='none'
#     )

#     img.seek(0)

#     plot_url = base64.b64encode(
#         img.getvalue()
#     ).decode('utf8')

#     plt.close(fig)

#     items = [
#         {
#             "name": genre,
#             "value": f"{count} Tracks"
#         }
#         for genre, count in data
#     ]

#     return render_template(
#         'data_list.html',
#         title="Genre Statistics",
#         items=items,
#         plot_url=plot_url
#     )

# @app.route('/ping')
# def ping():

#     response = influxdb.connection.ping()

#     return {
#         "connected": response
#     }

# @app.route('/buckets')
# def buckets():

#     buckets_api = influxdb.connection.buckets_api()

#     buckets = buckets_api.find_buckets().buckets

#     return {
#         "buckets": [
#             bucket.name
#             for bucket in buckets
#         ]
#     }

# @app.route('/weather')
# def weather():

#     query_api = influxdb.connection.query_api()

#     org = app.config["INFLUXDB_V2_ORG"]

#     flux_query = '''
#     from(bucket: "weather")
#       |> range(start: -1w)
#       |> filter(fn: (r) => r._measurement == "weather_current")
#       |> filter(fn: (r) => r._field == "temperature_2m")
#       |> filter(fn: (r) => r.location == "annecy")
#     '''

#     tables = query_api.query(
#         org=org,
#         query=flux_query
#     )

#     data = []
#     time = []

#     for table in tables:

#         for record in table.records:

#             data.append(
#                 record.values["_value"]
#             )

#             time.append(
#                 str(record.values["_time"])
#             )

#     return {
#         "data": data,
#         "time": time
#     }

# @app.route('/measurements')
# def measurements():

#     query_api = influxdb.connection.query_api()

#     org = app.config["INFLUXDB_V2_ORG"]

#     flux_query = '''
#     import "influxdata/influxdb/schema"

#     schema.measurements(bucket: "weather")
#     '''

#     tables = query_api.query(
#         org=org,
#         query=flux_query
#     )

#     meas_list = [
#         record.values["_value"]
#         for table in tables
#         for record in table.records
#     ]

#     return {
#         "measurements": meas_list
#     }

# @app.route('/fields')
# def fields():

#     query_api = influxdb.connection.query_api()

#     org = app.config["INFLUXDB_V2_ORG"]

#     flux_query = '''
#     import "influxdata/influxdb/schema"

#     schema.fieldKeys(bucket: "weather")
#     '''

#     tables = query_api.query(
#         org=org,
#         query=flux_query
#     )

#     field_list = [
#         record.values["_value"]
#         for table in tables
#         for record in table.records
#     ]

#     return {
#         "fields": field_list
#     }

# def format_dataset(data, x):

#     X = []
#     y = []

#     for i in range(len(data) - 2 * x + 1):

#         X.append(data[i:i + x])
#         y.append(data[i + x:i + 2 * x])

#     return X, y

# @app.route('/format')
# def format_test():

#     data = [
#         1, 2, 3, 4,
#         5, 6, 7, 8
#     ]

#     x = 3

#     X, y = format_dataset(
#         data,
#         x
#     )

#     return {
#         "X": X,
#         "y": y
#     }

# if __name__ == '__main__':

#     app.run(
#         host='0.0.0.0',
#         port=5047,
#         debug=True
#     )





