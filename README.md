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

## PW1 – Random Forest with scikit-learn

Page: `http://localhost:5004/rf/`

### Goal
1. Integrate the Random Forest classifier from the subject into a Flask route, with the choice of the dataset (iris, diabetes or digits) and of the hyperparameters.
2. For each dataset, study the score as a function of the hyperparameters and comment on the results.

### How to use
1. Choose a dataset and the hyperparameters:
   - `test_size`: part of the data kept for the test (0.05–0.9);
   - `n_estimators`: number of trees in the forest (1–500);
   - `max_depth`: maximum depth of each tree (empty = no limit);
   - `random_state`: seed, so that the same parameters always give the same result.
2. **Train model** shows the score on the test set and, for classification, the classification report.
3. **Sensitivity study** trains the model for every combination of `test_size` (0.1–0.5) and `n_estimators` (1, 10, 50, 100) and shows the scores in a table.

### Files
| File | Role |
|---|---|
| `app/services/random_forest.py` | Loading the datasets, training, evaluation and sensitivity study |
| `app/routes/random_forest.py` | `/rf` route: reads and validates the form, calls the service |
| `app/templates/rf_dashboard.html` | Form, result and sensitivity table |

### Deviations from the instructions

Some parts of the code given in the subject do not work as written, or give misleading results. Here is what we changed and why.

| # | Instruction / code of the subject | Problem | What we did |
|---|---|---|---|
| 1 | Use iris, diabetes **and** digits with `RandomForestClassifier` and report the accuracy | `diabetes` is a **regression** dataset: the target is a continuous value (disease progression). There are 214 different values for 442 samples, so a classifier treats each value as a separate "class". We tested it: the accuracy is **0.0**, and scikit-learn itself warns that `y` could represent a regression problem. | For diabetes we use `RandomForestRegressor` and evaluate it with the **R²** score (and the MSE) instead of the accuracy. |
| 2 | `metrics.classification_report(ypred, ytest)` | The signature is `classification_report(y_true, y_pred)`. With the arguments swapped, precision and recall are exchanged in the report. | Fixed to `classification_report(ytest, ypred)`. |
| 3 | `train_test_split(...)` and `RandomForestClassifier(...)` without `random_state` | Every run gives another split and other trees, so the results change each time. On iris, 5 runs with the same parameters gave 0.93, 0.93, 0.93, 0.80 and 1.00. A sensitivity study is meaningless if the score also changes for random reasons. | `random_state` is fixed (42 by default) and can be changed in the form, so only the studied hyperparameter changes. |
| 4 | Simple split, without stratification | With small datasets, a class can be under-represented in the test set (the course, section 3.1, presents stratified sampling for this reason). | `stratify=y` is used for the classification datasets (iris, digits). It is not possible for diabetes, whose target is continuous. |
| 5 | `print(digits.keys())` | A `print` in a web route only writes to the server console, the user does not see it. | The page shows the information about the dataset instead: number of samples, features, train and test sizes. |
| 6 | `test_size = 0.1` on iris | Not a bug, but iris has only 150 samples, so the test set has only 15 flowers: one error changes the accuracy by 6.7 %. | Kept, and explained in the analysis of the results below. |

## Security
- Do not commit `.env` files or API tokens.
- Keep credentials out of source code.
- Do not expose Flask debug mode to untrusted networks.

## Academic Context

This project demonstrates Flask web development, database integration, InfluxDB time-series queries, data visualisation, and dataset formatting.
