# Customer clustering on the Chinook database (K-Means)

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func

from extensions import db
from models.chinook import Customer, Genre, Invoice, InvoiceLine, Track

FEATURES = ["total_spent", "n_invoices", "n_genres", "favorite_share"]
FEATURE_LABELS = {
    "total_spent": "Total spent ($)",
    "n_invoices": "Invoices",
    "n_genres": "Different genres",
    "favorite_share": "Favourite genre share (%)",
}
K_MIN, K_MAX = 2, 10


def get_customer_features():
    """One line per customer: spending, invoices and favourite genre."""
    # total spent and number of invoices
    spending = (
        db.session.query(
            Customer.CustomerId,
            Customer.FirstName,
            Customer.LastName,
            func.sum(Invoice.Total),
            func.count(Invoice.InvoiceId),
        )
        .join(Invoice, Invoice.CustomerId == Customer.CustomerId)
        .group_by(Customer.CustomerId)
        .all()
    )

    # number of tracks bought in each genre
    genres = (
        db.session.query(
            Invoice.CustomerId,
            Genre.Name,
            func.sum(InvoiceLine.Quantity),
        )
        .select_from(Invoice)
        .join(InvoiceLine, InvoiceLine.InvoiceId == Invoice.InvoiceId)
        .join(Track, Track.TrackId == InvoiceLine.TrackId)
        .join(Genre, Genre.GenreId == Track.GenreId)
        .group_by(Invoice.CustomerId, Genre.GenreId)
        .all()
    )

    tracks_per_genre = {}
    for customer_id, genre, quantity in genres:
        tracks_per_genre.setdefault(customer_id, {})[genre] = int(quantity)

    customers = []
    for customer_id, first_name, last_name, total, n_invoices in spending:
        counts = tracks_per_genre.get(customer_id, {})
        n_tracks = sum(counts.values())
        favorite = max(counts, key=counts.get) if counts else "-"
        customers.append({
            "id": customer_id,
            "name": f"{first_name} {last_name}",
            "total_spent": round(float(total), 2),
            "n_invoices": int(n_invoices),
            "n_genres": len(counts),
            "favorite_genre": favorite,
            "favorite_share": round(100 * counts.get(favorite, 0) / n_tracks, 1) if n_tracks else 0.0,
        })
    return customers


def _scaled_matrix(customers):
    X = np.array([[c[f] for f in FEATURES] for c in customers], dtype=float)
    # the features have different units ($, counts, %), so they are scaled first
    return StandardScaler().fit_transform(X)


def elbow_curve(customers, k_max=K_MAX, random_state=42):
    """Inertia for k = 1..k_max, to choose k with the elbow method."""
    X = _scaled_matrix(customers)
    ks = list(range(1, min(k_max, len(customers)) + 1))
    inertias = [
        KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X).inertia_
        for k in ks
    ]
    return ks, inertias


def cluster_customers(customers, k=3, random_state=42):
    """Run K-Means and return the label of every customer and a summary per cluster."""
    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"k must be between {K_MIN} and {K_MAX}")

    X = _scaled_matrix(customers)
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)

    for customer, label in zip(customers, labels):
        customer["cluster"] = int(label)

    summary = []
    for cluster in range(k):
        members = [c for c in customers if c["cluster"] == cluster]
        genres = [c["favorite_genre"] for c in members]
        summary.append({
            "cluster": cluster,
            "size": len(members),
            "total_spent": round(float(np.mean([c["total_spent"] for c in members])), 2),
            "n_invoices": round(float(np.mean([c["n_invoices"] for c in members])), 2),
            "n_genres": round(float(np.mean([c["n_genres"] for c in members])), 1),
            "favorite_share": round(float(np.mean([c["favorite_share"] for c in members])), 1),
            "main_genre": max(set(genres), key=genres.count),
        })
    return customers, summary


def _to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _dark_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#242424")
    ax.set_facecolor("#242424")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    return fig, ax


def elbow_plot(ks, inertias, k=None):
    fig, ax = _dark_axes((6, 4))
    ax.plot(ks, inertias, marker="o", color="#dc3545")
    if k in ks:
        ax.plot(k, inertias[ks.index(k)], marker="o", markersize=12,
                markerfacecolor="none", markeredgecolor="white")
    ax.set_xlabel("k (number of clusters)", color="white")
    ax.set_ylabel("Inertia", color="white")
    ax.set_title("Elbow method", color="white")
    fig.tight_layout()
    return _to_base64(fig)


def scatter_plot(customers, x="total_spent", y="favorite_share"):
    fig, ax = _dark_axes((6, 4))
    for cluster in sorted({c["cluster"] for c in customers}):
        members = [c for c in customers if c["cluster"] == cluster]
        ax.scatter([c[x] for c in members], [c[y] for c in members],
                   label=f"Cluster {cluster}", s=40)
    ax.set_xlabel(FEATURE_LABELS[x], color="white")
    ax.set_ylabel(FEATURE_LABELS[y], color="white")
    ax.set_title("Customers by cluster", color="white")
    ax.legend(facecolor="#242424", labelcolor="white")
    fig.tight_layout()
    return _to_base64(fig)
