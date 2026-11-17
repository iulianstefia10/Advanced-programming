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

### Random Forest (PW1)
- Train a random forest on the iris, digits or diabetes dataset.
- Change the hyperparameters from the web page.
- Run a sensitivity study of the score as a function of the hyperparameters.

## Technologies
- Python
- Flask
- scikit-learn
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

## PW1 – Random Forest

Page: `http://localhost:5004/rf/`

We added a page where you choose a dataset (iris, digits or diabetes), set the hyperparameters (`test_size`, `n_estimators`, `max_depth`, `random_state`) and train a random forest.
The **Sensitivity study** button trains the model for several values of `test_size` and `n_estimators` and shows the results in a table.

Files:
- `app/services/random_forest.py` – training and sensitivity study
- `app/routes/random_forest.py` – the `/rf` route
- `app/templates/rf_dashboard.html` – the page

### What we changed compared to the subject

- **diabetes is a regression dataset.** With `RandomForestClassifier` the accuracy was 0, so for diabetes we use `RandomForestRegressor` and the R² score.
- **`classification_report(ypred, ytest)`** had the arguments in the wrong order, we use `classification_report(ytest, ypred)`.
- **No `random_state`** in the subject, so the results changed at every run. We fixed it to 42 (it can be changed in the form).
- We added **`stratify`** to the split for iris and digits, so every class is in the test set.

### Results

Best results we got (100 trees, test_size = 0.1):

| Dataset | Score |
|---|---|
| digits | accuracy 0.97 |
| iris | accuracy 0.93 |
| diabetes | R² 0.53 |

What we noticed:
- One tree is much worse than a forest (digits: 0.76 with 1 tree, 0.97 with 10 trees). After about 50 trees the score does not improve much.
- A bigger `test_size` leaves less data for training, so the score goes down a little.
- iris is very small (15 test flowers with test_size = 0.1), so its results change a lot between combinations.
- diabetes is harder: the R² stays around 0.4–0.5.

### Tests

```bash
python -m unittest discover -s tests -v
```

The tests check the service (metric, sizes, random_state, stratify) and the `/rf` page (form, training, errors).

## Security
- Do not commit `.env` files or API tokens.
- Keep credentials out of source code.
- Do not expose Flask debug mode to untrusted networks.

## Academic Context

This project demonstrates Flask web development, database integration, InfluxDB time-series queries, data visualisation, and dataset formatting.
