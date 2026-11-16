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

### Sensitivity study

Score on the test set for every combination of `test_size` and `n_estimators` (`random_state = 42`, no `max_depth` limit). The same tables can be obtained with the **Sensitivity study** button.

**digits** (1,797 images 8×8, 10 classes) – accuracy

| test_size \ n_estimators | 1 | 10 | 50 | 100 |
|---|---|---|---|---|
| 0.1 | 0.7611 | 0.9722 | 0.9722 | 0.9722 |
| 0.2 | 0.7917 | 0.9361 | 0.9583 | 0.9611 |
| 0.3 | 0.7704 | 0.9426 | 0.9593 | 0.9667 |
| 0.4 | 0.7497 | 0.9332 | 0.9611 | 0.9666 |
| 0.5 | 0.7620 | 0.9288 | 0.9511 | 0.9577 |

**iris** (150 flowers, 4 features, 3 classes) – accuracy

| test_size \ n_estimators | 1 | 10 | 50 | 100 |
|---|---|---|---|---|
| 0.1 | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| 0.2 | 0.9667 | 0.9667 | 0.9000 | 0.9000 |
| 0.3 | 0.8667 | 0.9111 | 0.8889 | 0.8889 |
| 0.4 | 0.9333 | 0.9500 | 0.9333 | 0.9167 |
| 0.5 | 0.8800 | 0.8800 | 0.8933 | 0.8933 |

**diabetes** (442 patients, 10 features, regression) – R²

| test_size \ n_estimators | 1 | 10 | 50 | 100 |
|---|---|---|---|---|
| 0.1 | 0.1724 | 0.4529 | 0.5122 | 0.5252 |
| 0.2 | 0.1431 | 0.4082 | 0.4254 | 0.4428 |
| 0.3 | -0.0548 | 0.4016 | 0.4569 | 0.4703 |
| 0.4 | -0.1665 | 0.3935 | 0.4593 | 0.4888 |
| 0.5 | -0.1329 | 0.3716 | 0.3932 | 0.4074 |

### Analysis

- **Number of trees (`n_estimators`).** This is the most important hyperparameter. A single tree is much weaker than a forest: on digits the accuracy goes from about 0.76 with 1 tree to 0.93–0.97 with 10 trees, and on diabetes the R² goes from 0.17 (or even negative) to about 0.45. Averaging many trees reduces the overfitting of each individual tree. After 50 trees the gain becomes small (plateau), while the training time keeps growing.
- **Size of the test set (`test_size`).** A bigger test set means less data for training, so the score generally decreases: on diabetes with 100 trees the R² goes from 0.53 (test 10 %) to 0.41 (test 50 %), on digits from 0.97 to 0.96. Digits is less affected because it has more samples.
- **iris.** The problem is easy (the classes are almost separable), so even one tree reaches about 0.93. The differences in the table are mostly noise: with 15 to 75 test flowers, one or two errors change the accuracy by several percent, which is why more trees do not always give a better score here.
- **diabetes.** The R² stays around 0.4–0.5: the 10 features explain only about half of the variance of the disease progression, it is a harder problem. A negative R² with a single tree means that this tree predicts worse than simply giving the mean value: it overfits the training data.
- **Conclusion.** For these datasets, 50 to 100 trees and a test set of 20–30 % are a good compromise. The results only make sense with a fixed `random_state`, and with the right type of model (regression for diabetes).

## Security
- Do not commit `.env` files or API tokens.
- Keep credentials out of source code.
- Do not expose Flask debug mode to untrusted networks.

## Academic Context

This project demonstrates Flask web development, database integration, InfluxDB time-series queries, data visualisation, and dataset formatting.
