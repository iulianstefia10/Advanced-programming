# Advanced Programming

A Flask web application developed for the R5.A.05 Advanced Programming practical exercises at IUT Annecy.

## Features

### Chinook Database
- Browse artists and albums.
- Explore relationships between artists and albums.
- View statistics about tracks, genres, sales, and customers.
- Inspect database tables and columns.

### InfluxDB
- Check the InfluxDB connection status.
- Browse available buckets, measurements, and fields.
- Retrieve weather time-series data for Annecy.
- Visualise weather measurements.
- Format time-series data into input windows and prediction targets.

## Technologies
- Python
- Flask
- Flask-SQLAlchemy
- SQLite
- InfluxDB 2.x
- HTML, CSS, and JavaScript
- Docker and Docker Compose

## Project Structure

```text
Advanced-Programming/
├── app/
│   ├── app.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── routes/
│   ├── templates/
│   └── static/
│       └── css/
├── .env
└── README.md
```

## Running the Application

Configure the required environment variables in `.env`, including the InfluxDB connection settings and API token.

From the project root, build and start the application:

```bash
docker compose --env-file .env -f app/docker-compose.yml up -d --build flask-app
```

The application is configured to use port `5004` on the host.

Open the application at:

```text
http://localhost:5004/
```

## Security
- Do not commit `.env` files or API tokens.
- Keep credentials out of source code.
- Do not expose Flask debug mode to untrusted networks.

## Academic Context

This project demonstrates Flask web development, database integration, InfluxDB time-series queries, data visualisation, and dataset formatting.
