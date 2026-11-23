# Advanced Programming

A Flask web application developed for the R5.A.05 Advanced Programming practical exercises at IUT Annecy.

The application is a "DataHub": one home page and one page for every exercise of the subject (databases, time series, classical machine learning, neural networks, inference of pre-trained models). Two databases are used: **SQLite** (the Chinook demo database) and **PostgreSQL** (our own tables, for example the digit recognition models).

## Pages

| Page | URL | Subject |
|---|---|---|
| Home | `/` | – |
| Chinook database | `/chinook/` | Sessions 2-3, Ex 1 (+ customer clusters, `/chinook/clusters`) |
| InfluxDB | `/influxdb/` | Sessions 2-3, Ex 2 |
| Time-series forecasting | `/forecast/` | Sessions 2-3, Ex 2 (§7.4), extended |
| Apparent temperature | `/apparent/` | Session 1, §5.4 |
| Digit recognition (models in PostgreSQL) | `/mnist/` | Sessions 2-3, Ex 3 (§8.2 to §8.4) |
| Random Forest | `/rf/` | PW1 |
| Neural network on MNIST | `/mlp/` | PW2 |
| CNN on CIFAR-100 | `/cnn/` | PW3 |
| PyTorch on CPU, GPU and TPU (report) | `/pw4/` | PW4 |
| Land cover from a satellite image | `/landcover/` | PW5 |
| Object detection (YOLO) | `/yolo/` | – |

## Features

### Chinook Database
- Browse artists and albums.
- Explore relationships between artists and albums.
- View statistics about tracks, genres, sales, and customers.
- Inspect database tables and columns.
- Group the customers into clusters with K-Means (`/chinook/clusters`).

### InfluxDB
- Check the InfluxDB connection status.
- Browse available buckets, measurements, and fields.
- Retrieve weather time-series data for Annecy.
- Visualise weather measurements.
- Format time-series data into input windows and prediction targets.

### Time-series forecasting
- Forecast the temperature of Annecy from several variables (temperature, humidity, wind…) of several nearby stations (`/forecast/`).
- Compare five models: naive and seasonal baselines, linear regression, MLP and LSTM, with MAE and RMSE on unseen data.
- See how much the other variables and cities help, and the forecast of every model on a chart.
- The same page works on the `crypto` and `forex` buckets.

### Apparent temperature (Session 1)
- Compute the felt temperature from temperature, humidity and wind (heat index, wind chill, Steadman).
- Load the current weather of a station from InfluxDB.

### Digit recognition (Session 3)
- Draw a digit on the page or upload an image.
- Choose one stored model or all of them and compare the predictions, with the probabilities.
- The digit models are **stored in the PostgreSQL database**: import a `.h5` or `.keras` model from the page, delete it, or select it for the inference.

### Object detection (YOLO)
- Upload a photo and find the objects with a small ready YOLO model (80 COCO classes).
- See the boxes, the labels and the confidence.

### Random Forest (PW1)
- Train a random forest on the iris, digits or diabetes dataset.
- Change the hyperparameters from the web page.
- Run a sensitivity study of the score as a function of the hyperparameters.

### CIFAR-100 CNN (PW3)
- Train a convolutional neural network on CIFAR-100 (the dataset is imported by the page).
- Change the hyperparameters from the web page (fine or coarse labels, filters, kernel size, optimizer...).
- Follow the download and the training live, and stop them with a button.

### Neural network on MNIST (PW2)
- Train a fully connected network (MLP) on MNIST and change its architecture and training hyperparameters.
- Run a sensitivity study of the accuracy.

### PyTorch on CPU, GPU and TPU (PW4)
- Google Colab notebook comparing the training and inference times of the same network on CPU, GPU and TPU.
- Report page `/pw4/` with the results, and an Inference button to test our exported model and a pretrained model on one image.

### Land cover from a satellite image (PW5)
- Google Colab notebook that trains a VGG16 classifier on the EuroSAT dataset (RGB, 10 land cover classes).
- Upload a satellite screenshot (for example from Google Earth): the image is cut into patches, every patch is classified and the land cover map is drawn (`/landcover/`).
- Download the map as a `.npy` matrix.

## Technologies
- Python 3.12, Flask, Jinja2
- Flask-SQLAlchemy with two databases: SQLite (Chinook) and PostgreSQL 15 (our own tables, `psycopg2`)
- InfluxDB 2.x and the Flux language (time series)
- scikit-learn (random forest, K-Means, linear regression)
- TensorFlow / Keras (MLP, CNN, LSTM, VGG16 for the land cover)
- OpenCV and ONNX (YOLO object detection)
- NumPy and pandas
- PyTorch in Google Colab (PW4)
- HTML, CSS (Bootstrap) and JavaScript (charts drawn in SVG)
- Docker and Docker Compose (the Flask application and the PostgreSQL service)
- GitHub Actions (continuous integration)

## Project Structure

```text
Advanced-Programming/
├── .github/workflows/ci.yml      # tests + Docker build on every push (Session 7)
├── app/
│   ├── app.py                    # creates the Flask app, registers every page
│   ├── extensions.py             # SQLAlchemy object
│   ├── Dockerfile
│   ├── docker-compose.yml        # Flask application + PostgreSQL service (db)
│   ├── database/chinook.db       # SQLite database (Sessions 2-3, Ex 1)
│   ├── models/                   # SQLAlchemy models
│   │   └── app_data.py           # stored_model: the digit models of /mnist/ (PostgreSQL)
│   ├── routes/                   # one blueprint per page
│   │   ├── chinook.py            # /chinook/...          Sessions 2-3, Ex 1 + clusters
│   │   ├── influxdb.py           # /influxdb/...         Sessions 2-3, Ex 2
│   │   ├── forecast.py           # /forecast/            time-series forecasting
│   │   ├── apparent.py           # /apparent/            Session 1, 5.4
│   │   ├── mnist.py              # /mnist/               Sessions 2-3, Ex 3
│   │   ├── random_forest.py      # /rf/                  PW1
│   │   ├── mlp.py                # /mlp/                 PW2
│   │   ├── cnn.py                # /cnn/                 PW3
│   │   ├── landcover.py          # /landcover/           PW5
│   │   └── yolo.py               # /yolo/                object detection
│   ├── services/                 # the models and the computations of each page
│   │   └── model_store.py        # digit models in the database (first copy, import)
│   ├── utils/dataset.py          # windows for time series (format_dataset, make_windows)
│   ├── templates/                # one page per feature + base.html and home.html
│   ├── static/                   # CSS, uploaded images (not committed)
│   ├── models_store/             # PW5 model: eurosat_vgg16.keras and class_indices.npy
│   ├── MNIST_keras_CNN.h5        # given CNN (Ex 3), copied into the database at the first start
│   ├── mnist_mlp.keras           # our PW2 MLP, saved by train_mnist_mlp.py (also copied into the database)
│   ├── yolov8n.onnx              # YOLO model
│   ├── pw2_sensitivity.py        # PW2 results for this README
│   └── forecast_report.py        # forecasting results for this README
├── notebooks/                    # Google Colab notebooks (PW4, PW5)
├── tests/                        # unit tests: python -m unittest discover -s tests -v
├── requirements.txt
├── .env.example                  # copy to .env and fill the InfluxDB token
└── README.md
```

## Running the Application

Configure the required environment variables in `.env`, including the InfluxDB connection settings and API token.
Copy `.env.example` to `.env` and fill in `INFLUXDB_V2_TOKEN`. If the InfluxDB server gets a new address (for example after a school network change), only `INFLUXDB_V2_URL` in `.env` has to be changed. When the server can't be reached, the InfluxDB pages show a short "InfluxDB is unavailable" message instead of an error page.

From the project root, build and start the application:

```bash
docker compose --env-file .env -f app/docker-compose.yml up -d --build flask-app
```

The application is configured to use port `5004` on the host.

Open the application at:

```text
http://localhost:5004/
```

### PostgreSQL

The `db` service (PostgreSQL 15) starts together with the application: Flask waits until it is healthy (`depends_on` with a healthcheck), and the data is kept in the `pgdata` volume, so it stays after the containers are rebuilt.
The address comes from `DATABASE_URL`, built in `docker-compose.yml`. The password can be set with `POSTGRES_PASSWORD` in `.env` (default for local work: `password123`).
Without `DATABASE_URL` (tests, continuous integration, running without Docker) our own tables are created in a local SQLite file, `app/database/appdata.db`, so nothing else has to be installed.

To look at the data:

```bash
docker compose -f app/docker-compose.yml exec db psql -U group4 -d flask_models -c "SELECT id, name, filename, size_bytes FROM stored_model;"
```

## Session 1 – Apparent temperature

Page: `http://localhost:5004/apparent/`

The first page asked in Session 1 (§5.4): a Python model that computes the apparent ("feels like") temperature from inputs the user can change, like [alpiniste.fr](https://www.alpiniste.fr/indice-chaleur-calculateur/).
You enter the air temperature, the humidity and the wind, or press **Use the current weather** to load the last values of a station from InfluxDB (`weather_current`, Annecy by default).

Files:
- `app/services/apparent.py` – the three formulas, the choice of formula, the checks and the InfluxDB query
- `app/routes/apparent.py` – the `/apparent` route
- `app/templates/apparent_dashboard.html` – the page

### Model

One formula is not valid for every weather, so the app picks the right one:

| Conditions | Formula | Why it feels different |
|---|---|---|
| T ≥ 27 °C and humidity ≥ 40 % | Heat index (NOAA, Rothfusz regression) | sweat evaporates less in humid air |
| T ≤ 10 °C and wind > 4.8 km/h | Wind chill (Environment Canada) | the wind removes heat from the skin |
| any other case | Steadman apparent temperature (Australian Bureau of Meteorology) | uses temperature, humidity and wind together |

The page shows the value, the formula used, the three formulas side by side, a comfort level and how the felt temperature changes with more wind.

### What we paid attention to

- The **heat index formula is written in °F**: applied directly to °C it gives wrong values. We convert °C → °F → °C.
- The heat index is **only valid in hot and humid air**; used in the cold it gives meaningless numbers. Same for the wind chill above 10 °C.
- The wind chill uses the wind **in km/h at 10 m**, Steadman uses it **in m/s**.
- The **heat index ignores the wind**: at 30 °C it gives the same value with or without wind. The chart "same air with more wind" therefore uses the Steadman model in warm weather and the wind chill in the cold.
- Wrong values (letters, humidity above 100 %, negative wind…) and an unreachable InfluxDB show an error message instead of crashing the page.

The formulas were checked against the NOAA heat index table and the Environment Canada wind chill table (difference below 1 °F / 0.5 °C).

### Tests

```bash
python -m unittest tests.test_apparent_service tests.test_apparent_route -v
```

## Chinook – Customer clusters (K-Means)

Page: `http://localhost:5004/chinook/clusters`

For every customer we compute, with SQLAlchemy queries:
- the total spent (sum of the invoices),
- the number of different genres bought,
- the favourite genre and its share of the tracks bought (%),
- the number of invoices (shown in the tables only, see below).

The three numbers are scaled with `StandardScaler` (they have different units) and grouped with K-Means.
`k` is chosen in the form (2 to 10). The page shows the elbow curve, a scatter plot coloured by cluster and the table of the clusters and of the customers.

Files:
- `app/services/chinook_clusters.py` – queries, K-Means, elbow curve and plots
- `app/routes/chinook.py` – the `/chinook/clusters` route
- `app/templates/chinook_clusters.html` – the page

### What we noticed

- Chinook is a demo database: 58 of the 59 customers have exactly 7 invoices. With the number of invoices as a fourth feature, the only customer with 6 invoices (Puja Srivastava) was about 7.6 standard deviations from the others after scaling and always got a cluster of her own (k = 3 gave 25 / 33 / **1**). So this number is no longer given to K-Means.
- Without it the elbow is at **k = 3**: the inertia goes 177 → 104 → 65, then decreases slowly (54, 44...). The silhouette score is also better (0.39 instead of 0.35).
- With k = 3: 23 customers who mostly buy their favourite genre (51 % of their tracks, 6 genres), 9 big spenders who buy many genres (45.3 $, 9.3 genres) and 27 "average" customers (38.6 $, 8.1 genres).
- Rock is the favourite genre of 47 customers, so the favourite genre is more useful as a share (%) than as a name.
- `random_state = 42` and `n_init = 10` make the result the same at every run.

### Tests

```bash
python -m unittest tests.test_chinook_clusters -v
```

## Time-series forecasting (InfluxDB weather, several variables and cities)

Page: `http://localhost:5004/forecast/`

Sessions 2–3 (Exercise 2, §7.4) prepared the data: read a weather series from InfluxDB and cut it into windows of x points to predict the x next points. As asked by the teacher, we use these windows to **forecast the temperature of Annecy from several variables and several cities around Annecy**, and we compare five models.

Files:
- `app/services/forecast.py` – reading InfluxDB, hourly table, windows, split, the five models, the errors
- `app/routes/forecast.py` – the `/forecast` route
- `app/templates/forecast_dashboard.html` – the page and its two charts
- `app/utils/dataset.py` – `make_windows`, the version of `format_dataset` with several variables
- `app/forecast_report.py` – prints the results below from the real InfluxDB data

### How it works

1. **What exists in the bucket** is read with the Flux `schema` functions: measurements, stations (tag `location`), variables (fields). The form only accepts these values, so nothing typed by the user goes into the Flux query.
2. **One value per hour** for every chosen variable of every chosen station (`aggregateWindow(every: 1h)`), put in one table with pandas; holes of up to 3 hours are interpolated, series with too many missing hours are left out.
3. **Windows**: the last x hours of all the series are the input, the next hours of the target are the output (`make_windows`; with one variable it gives the same windows as `format_dataset`). Each series is given **relative to its last value** in the window, and the models predict the **change** from the last known temperature.
4. **Split by time, never shuffled**: the first 80 % of the windows to train, the last 20 % to test, with a gap between them so that no hour is used both for training and for testing. The scaling (mean and standard deviation) is computed on the training hours only.
5. **Five models**:

| Model | Idea |
|---|---|
| Naive | the next hours equal the last known value |
| Seasonal naive | the next hours equal the same hours of the previous day (temperature has a 24 h cycle) |
| Linear regression | ridge regression on the target window and the last 6 hours of the other series; the strength of the regularisation is chosen by time |
| MLP | dense network 64 → 32 → horizon with dropout and L2 (TensorFlow), same inputs as the linear model |
| LSTM | recurrent network with 32 units and dropout that reads the whole window of every series hour by hour (TensorFlow) |

The MLP and the LSTM stop when the error on the most recent 10 % of the training windows stops improving (early stopping, patience 5).

6. **Errors** on the test period, in °C: MAE (mean absolute error), RMSE, and the MAE for every forecast hour (+1 h … +x h). Every learned model is trained twice, **with all the inputs** and **with the Annecy temperature only**; the column "gain" shows how much the other variables and cities help.

### What we paid attention to

- **No shuffling and a gap between train and test.** With overlapping windows, a random split puts almost the same hours in train and test and gives errors that are far too good.
- **Scaling on the training part only**, for the same reason.
- **Baselines first.** Temperature has a strong daily cycle; a model is only useful if it beats "same hour yesterday".
- The teacher's query reads **one week** (`range(start: -1w)`); that is about 170 hours, too few windows for an LSTM. The page reads 60 days by default (up to 365).
- **More inputs is not always better** (see the first run below): with few training windows, many inputs make the models overfit. Only the stations close to Annecy are ticked by default.
- `crypto` and `forex` have other tags and fields: the page reads them from the bucket, so the same code works on them.

### Results

#### First run (20.11, before the fixes): more inputs made it worse

30 days of data (668 hours, 451 training windows, 124 test windows), Annecy temperature, 24 h → next 24 h, with **35 input series** (temperature, humidity, cloud cover, pressure and wind of 7 stations, Bordeaux included). MAE and RMSE in °C.

| Model | MAE | RMSE | MAE, Annecy temperature only | Gain of the extra inputs |
|---|---|---|---|---|
| **Naive (last value)** | **2.14** | 2.71 | – | – |
| Seasonal naive (previous day) | 2.58 | 3.12 | – | – |
| Linear regression | 5.31 | 6.64 | 1.89 | −180 % |
| MLP | 6.68 | 8.91 | 2.61 | −156 % |
| LSTM | 3.25 | 3.92 | 2.56 | −27 % |

What it showed:
- With the **Annecy temperature only**, the linear regression (1.89 °C) already beats the naive forecast (2.14 °C).
- With all the inputs every learned model became **worse than the naive forecast**. 35 series × 24 hours = 840 numbers per window for only 451 training windows: the models learned the noise of the training period (**overfitting**). Far stations (Bordeaux is 600 km away) bring no information about the next hours in Annecy.
- The temperature drifts (colder in November than in October), so values given as absolute levels look different in the test period than in training.
- Crypto (`change_pct_24h` of ADAUSDT, 1 series): the linear regression was the best (MAE 2.56 vs 3.49 for the naive forecast). Forex (`rate` of USD): the naive forecast was the best (0.019); an exchange rate barely moves in 24 hours.

What we changed:
1. The models predict the **change** from the last known value, and every input series is given **relative to its last value**: a model that learns nothing gives the naive forecast instead of nonsense, and the drift of the level no longer matters.
2. **Fewer inputs** for the linear model and the MLP: the whole window of the target, but only the last 6 hours of the other series.
3. The **strength of the regularisation** of the ridge regression is chosen **by time** (train on the first 80 % of the training windows, validate on the last 20 %). The usual leave-one-out (`RidgeCV`) almost sees the answer with overlapping windows and chose a far too weak regularisation. Dropout and L2 for the MLP and the LSTM.
4. By default only the **stations close to Annecy** (Aix-les-Bains, Annemasse, Chambéry, Geneva, Rumilly) and three variables (humidity, pressure, wind) are ticked; 60 days of data instead of 30.

On a test with 30 useless drifting series added, the error of the linear model with all the inputs went from 2–3 times the error without them to about the same; the tests keep this (`RobustnessTests`).

#### After the fixes (20.11)

60 days of data (1,387 hours, 21.09 → 20.11.2026), Annecy and the 5 stations close to it, 4 variables (temperature, humidity, pressure, wind) = **24 input series**. MAE and RMSE in °C on the last 20 % (never seen in training). Output of `python forecast_report.py`.

| Model | 24 h → 24 h: MAE | RMSE | MAE, Annecy only | Gain | 24 h → 6 h: MAE | RMSE | MAE, Annecy only | Gain |
|---|---|---|---|---|---|---|---|---|
| Naive (last value) | 2.58 | 3.32 | – | – | 1.51 | 2.14 | – | – |
| Seasonal naive (previous day) | 2.55 | 3.16 | – | – | 2.59 | 3.19 | – | – |
| Linear regression | **2.09** | 2.65 | 2.18 | +4.1 % | 1.27 | 1.68 | 1.24 | −2.6 % |
| MLP | **2.09** | **2.63** | 2.09 | +0.2 % | **1.18** | **1.55** | 1.14 | −3.1 % |
| LSTM | 2.17 | 2.77 | 2.26 | +3.7 % | 1.28 | 1.75 | 1.29 | +1.3 % |

Other buckets (one series each, 24 h → 24 h):

| Model | Crypto: `price` of BTCUSDT (MAE, $) | Forex: `rate` of USD (MAE) |
|---|---|---|
| Naive (last value) | **832** | **0.025** |
| Seasonal naive | 1,270 | 0.047 |
| Linear regression | 1,056 | 0.026 |
| MLP | 853 | 0.109 |
| LSTM | 921 | 0.056 |

What we noticed:
- **The fixes changed everything.** Before them the learned models were worse than the naive forecast; now all three beat both baselines: for the next 24 h, 2.09 °C instead of 2.55 °C (−19 %); for the next 6 h, the MLP makes 1.18 °C instead of 1.51 °C (−22 %).
- **Other cities and variables help a little for the next day** (+4 % for the linear model and the LSTM) and not for the next 6 hours (−3 %): in the short term the recent temperature of Annecy itself carries almost all the information. The big gain comes from the way the data is given to the models, not from adding stations; without the fixes, adding them made the error 2–3 times bigger.
- **More data helped**: 60 days give 1,025 training windows instead of 451.
- The error grows with the forecast hour (chart "Error by forecast hour" on the page): +1 h is easy, +24 h is hard.
- **Crypto**: for the price of Bitcoin the naive forecast is the best (832 $), the MLP is close (853 $). Past prices hardly predict the next changes of a crypto price, which behaves almost like a random walk; it is the expected result, not a bug.
- **Forex**: the naive forecast is the best (0.025), the linear model is the same (0.026): an exchange rate barely moves in 24 hours.
- All the 182 tests of the project pass with these changes (`python -m unittest discover -s tests`).

### Tests

```bash
python -m unittest tests.test_forecast_service tests.test_forecast_route -v
```

The tests use a fake InfluxDB (two stations, one of them sees the weather 6 hours earlier) so they run without the server, also on GitHub. They check the windows, that train and test never share an hour, the Flux query, the form checks, that the linear model beats the naive baseline and that the other city helps. The MLP and LSTM test is skipped when TensorFlow is not installed.

## Session 3 – Digit recognition: draw a digit, CNN and MLP

Page: `http://localhost:5004/mnist/`

You can draw a digit with the mouse (or a finger on a phone) or upload an image.
The same prepared image is sent to the models you choose, and every prediction is shown with the probability of each digit. Two models are available from the first start, and more can be imported:
- **CNN**: `MNIST_keras_CNN.h5`, the model given in the subject,
- **MLP**: our PW2 network (128-64-32, pixels divided by 255), trained once with `app/train_mnist_mlp.py` and saved in `app/mnist_mlp.keras` (test accuracy 0.974).

Files:
- `app/services/mnist_inference.py` – preparing the image, loading a model from the database and running the models
- `app/services/model_store.py` – copies the two given models into the database at the first start, checks and stores an imported model
- `app/models/app_data.py` – the `stored_model` table
- `app/routes/mnist.py` – the `/mnist/` route (drawing or upload), the import and the deletion of models
- `app/templates/mnist_dashboard.html` – the canvas, the choice of the model, the table of models and the results
- `app/train_mnist_mlp.py` – trains and saves the MLP (run it again only if the model must change)

### What we paid attention to

- The two models do not take the same input: the CNN wants an image `(1, 28, 28, 1)`, the MLP wants a vector of **784** values. Both must be divided by **255**, like during the training.
- MNIST digits are **white on a black background**, but a drawing is black on white, so the image is inverted. A transparent background (canvas, PNG) is first made white.
- In MNIST the digit is about 20×20 pixels and centred in 28×28. Before, the whole picture was just resized, so a digit drawn in a corner or very big was badly recognised (a 7 was predicted as 1). Now the digit is cropped, fitted in a 20×20 box and moved so that its centre of mass is in the middle. The page shows this 28×28 image ("what the models see").
- With this preparation, on 200 MNIST test images turned into big black-on-white drawings, the CNN is right 199 times and the MLP 195 times. The CNN is a bit better on unusual shapes (for example a 6 written with a font).
- An uploaded file is saved in `static/uploads` only if it really is an image.
- `*.keras` files are ignored by git (PW3 models), so `app/mnist_mlp.keras` is added as an exception in `.gitignore`.

### Models in the database (Sessions 2-3, Ex 3, §8.3 and §8.4)

The subject asks to store the `.h5` models in a database and to choose, import and use them from the application. The models are therefore not read from files at run time: they are stored in the `stored_model` table of PostgreSQL.

| Column | Meaning |
|---|---|
| `id` | primary key |
| `name` | unique name shown in the page |
| `description` | optional text |
| `filename` | original name (gives the `.h5` or `.keras` extension) |
| `size_bytes` | size of the file |
| `data` | the model file (`LargeBinary`, `BYTEA` in PostgreSQL) |
| `created_at` | date of the import |

How it works:
1. **First start**: if the table is empty, `seed_default_models()` copies `MNIST_keras_CNN.h5` and `mnist_mlp.keras` into the database.
2. **Page**: a list lets you choose *All models* or one model; the same 28×28 image is sent to the chosen models.
3. **Import**: the file is written to a temporary folder (Keras loads from a file, not from bytes), loaded, and checked: it must take 28×28 pixels (784 values) and give 10 classes. Only then it is stored. A wrong file shows an error message instead of crashing the page.
4. **Inference**: a model is read from the database and loaded once, then kept in memory. The shape of the input (`(1, 28, 28, 1)` for the CNN, `(1, 784)` for the MLP) is read from the model itself, so an imported model works without any change in the code.
5. **Deletion**: the row is deleted and the model is removed from the memory.

Two databases are used by the application. The Chinook SQLite database stays the default one; our own tables use a second connection called `appdata` (`SQLALCHEMY_BINDS`), and `db.reflect(bind_key=None)` and `db.create_all(bind_key="appdata")` keep them apart. When PostgreSQL is not reachable, the application still starts and the page shows "The model database is unavailable".

What to know:
- An imported model gets pixels between 0 and 1. A model trained without the division by 255 will give wrong results.
- Loading a Keras file from a user can run code (especially `.h5`). It is acceptable for this course project, not for a public site.
- The images sent to the page are still saved in `static/uploads`, not in the database.
- The upload limit is 100 MB (`MAX_CONTENT_LENGTH`).
- The list of the page never loads the model files (`defer`), only the file of a model that is used.

### Tests

```bash
python -m unittest tests.test_mnist_inference -v
```

The tests that run the models need TensorFlow (they are skipped without it).

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

## PW2 – Neural network on MNIST (TensorFlow)

Page: `http://localhost:5004/mlp/`

We added a page where you choose the architecture (hidden layers and neurons, activation) and the training hyperparameters (`optimizer`, `loss`, `batch_size`, `epochs`, `validation_split`, normalization) and train a fully connected network (MLP) on MNIST.
The page shows the test accuracy, the test loss, the number of parameters, the training time and a chart of the accuracy per epoch.
The **Sensitivity study** button changes one hyperparameter (the others keep the values of the form) and shows the results in a table.

Files:
- `app/services/mlp.py` – checks, model, training and sensitivity study
- `app/routes/mlp.py` – the `/mlp` route
- `app/templates/mlp_dashboard.html` – the page
- `app/pw2_sensitivity.py` – runs the full study and prints the tables below

### What we changed compared to the subject

- **No normalization** in the subject: the pixels stay between 0 and 255, so the loss is huge. We divide the pixels by 255 (checkbox in the form, untick it to get the subject's code).
- **`model = Model(...)`** was written twice. It is not an error, we keep only one.
- **The loss must match the labels.** The labels are one-hot (`to_categorical`), so `sparse_categorical_crossentropy` would crash. The form only offers `categorical_crossentropy` and `mean_squared_error`.
- **No random seed**, so the results changed at every run. We use `tf.keras.utils.set_random_seed(42)`.
- `validation_split` takes the **last** part of the training data, without shuffling.
- Wrong values (`epochs = 0`, `batch_size = -5`, letters in the layers…) show an error message instead of crashing the page.

### Can we obtain better results?

Subject: test loss 2.7978, test accuracy 0.8993 (7,850 parameters).
Our runs: 60,000 training images, 20 epochs, adam, batch 128, validation_split 0.1.

| Model | Parameters | Test loss | Test accuracy |
|---|---|---|---|
| Subject (no hidden layer, no normalization) | 7,850 | 4.2349 | 0.8637 |
| Subject + normalization | 7,850 | 0.2617 | 0.9277 |
| 128-64-32 relu + normalization | 111,146 | 0.1309 | **0.9738** |

**Yes.** Normalization alone, with the same 7,850 parameters, divides the test loss by 16 and gives +6 points of accuracy.
Without it the training is unstable: we got 0.8637 here and 0.8786 with 5 epochs, against 0.8993 for the subject.
Adding three hidden layers with ReLU gives 0.9738, better than the subject (0.8993) and than the example of the PDF (0.9699).

### Sensitivity study

To keep it fast, the study uses the first 10,000 training images (the test set stays the full 10,000 images).
Base: layers 128,64,32, relu, adam, categorical_crossentropy, batch 128, 5 epochs, validation_split 0.1, normalization.
Only one hyperparameter changes in each table.

| activation | Test accuracy | Test loss |
|---|---|---|
| relu | **0.9295** | 0.2386 |
| tanh | 0.9277 | 0.2509 |
| sigmoid | 0.8812 | 0.5932 |

| Hidden layers | Parameters | Test accuracy | Test loss |
|---|---|---|---|
| none | 7,850 | 0.8910 | 0.4328 |
| 128 | 101,770 | 0.9321 | 0.2404 |
| 128,64 | 109,386 | **0.9323** | 0.2225 |
| 128,64,32 | 111,146 | 0.9295 | 0.2386 |

| Neurons | Parameters | Test accuracy | Test loss |
|---|---|---|---|
| decreasing 128→64→32 | 111,146 | **0.9295** | 0.2386 |
| increasing 32→64→128 | 36,842 | 0.9267 | 0.2467 |

| loss | Test accuracy | Test loss |
|---|---|---|
| categorical_crossentropy | 0.9295 | 0.2386 |
| mean_squared_error | **0.9310** | 0.0106 |

| optimizer | Test accuracy | Test loss |
|---|---|---|
| adam | **0.9295** | 0.2386 |
| rmsprop | 0.9277 | 0.2613 |
| sgd | 0.7862 | 0.8784 |

| batch_size | Test accuracy | Test loss | Time (s) |
|---|---|---|---|
| 32 | 0.9281 | 0.2447 | 5.1 |
| 128 | **0.9295** | 0.2386 | 2.4 |
| 512 | 0.9169 | 0.2936 | 1.7 |

| epochs | Test accuracy | Test loss | Time (s) |
|---|---|---|---|
| 5 | 0.9295 | 0.2386 | 2.7 |
| 10 | 0.9408 | **0.2216** | 4.0 |
| 20 | **0.9435** | 0.2623 | 7.7 |

| validation_split | Test accuracy | Test loss |
|---|---|---|
| 0.1 | **0.9295** | 0.2386 |
| 0.2 | 0.9276 | 0.2453 |
| 0.3 | 0.9282 | 0.2517 |

| Normalization | Test accuracy | Test loss |
|---|---|---|
| no | 0.8531 | 0.7501 |
| yes | **0.9295** | 0.2386 |

What we noticed:
- **Normalization** is the most important change: without it even the 3-layer model drops to 0.8531.
- **activation**: relu and tanh are almost equal, sigmoid is clearly worse (0.8812) because it learns slowly in 5 epochs.
- **Hidden layers**: the first hidden layer gives the big jump (0.8910 → 0.9321). More layers add parameters but no accuracy with 10,000 images and 5 epochs.
- **Neurons**: decreasing (128→64→32) is a bit better, but increasing (32→64→128) is close with 3 times fewer parameters, because the first layer (784 inputs) is the biggest one.
- **loss**: both losses give the same accuracy. Their values cannot be compared, they are not on the same scale.
- **optimizer**: adam and rmsprop adapt the step and are fast; sgd with its default learning rate is far behind after 5 epochs (0.7862).
- **batch_size**: 128 is the best compromise. 32 is twice as slow for no gain, 512 makes too few weight updates.
- **epochs**: accuracy still rises from 10 to 20 epochs, but the test loss goes up again (0.2216 → 0.2623): the model starts to overfit.
- **validation_split**: almost no effect between 0.1 and 0.3; a bigger split only leaves less data for training.

Best model: 128-64-32, relu, adam, batch 128, normalization, 20 epochs on all the data → **0.9738**.

### Role of each hyperparameter

| Hyperparameter | Role |
|---|---|
| activation | Adds non-linearity. Without it, several Dense layers work like one linear layer. |
| hidden layers / neurons | Size of the model. Too small underfits, too big is slow and can overfit. |
| loss | What the model minimises. It must match the format of the labels. |
| optimizer | How the weights are updated. adam and rmsprop adapt the step, sgd does not. |
| batch_size | Number of images per weight update. Small = more updates per epoch, big = faster epochs but fewer updates. |
| epochs | Number of passes over the training data. Too few underfits, too many overfits. |
| validation_split | Part of the training data kept aside to watch overfitting during the training. |
| normalization | Puts the pixels between 0 and 1, so the training is stable and converges. |

### Tests

```bash
python -m unittest discover -s tests -v
```

The tests check the form values, the sensitivity study, the number of parameters (7,850 and 111,146) and the `/mlp` page (form, training, errors).

## PW3 – CNN on CIFAR-100 (TensorFlow)

Page: `http://localhost:5004/cnn/`

We added a page where you set the hyperparameters of a convolutional neural network (`filters`, `kernel_size`, `dense_units`) and of the training (`epochs`, `batch_size`, `optimizer`, `learning_rate`, `validation_split`), choose fine or coarse labels, and train it on CIFAR-100 (60,000 colour images of 32x32 pixels, 100 classes).
The dataset is imported by the page itself. The training runs in the background: the page shows the download percentage, the epoch, the loss and the accuracy live, and a **Stop** button cancels the download or the training.
At the end it shows the test accuracy, the test loss, the number of parameters, the table of the metrics per epoch and the loss and accuracy curves (train and validation).

Files:
- `app/services/cnn_service.py` – dataset import, checks, model, training in a thread
- `app/routes/cnn.py` – the `/cnn` routes (`/`, `/train`, `/stop`, `/status`)
- `app/templates/cnn_dashboard.html` – the page
- `prepare_cifar100.py` – optional, prepares the dataset file before the first training

### What we changed compared to the subject

- **`cifar100.load_data()` blocks until the file is downloaded**, so a web page can neither show a percentage nor cancel it. We download the same archive ourselves (about 161 MB) and keep it in `app/data/cifar100.npz` (fine and coarse labels). The next trainings read this file, so there is no new download, even after the container is recreated. The model, the loss, the metrics and the division by 255 are those of the subject.
- **Only fine labels** in the subject. The form also offers the 20 superclasses (coarse labels): the last layer then has 20 neurons instead of 100.
- **The subject's code blocks until the end of the training**, which would freeze the page. We train in a background thread and the page asks `/cnn/status` regularly. Only one training can run at a time.
- **The architecture is a hyperparameter.** `filters` is the number of filters of the first convolution (doubled at each block: 32-64-128 by default), `dense_units` the size of the first dense layer (the second one has half of it). With the default values it is exactly the model of the subject: 270,372 parameters.
- **`kernel_size` is limited to 2–4.** With three convolutions without padding and three poolings on 32x32 images, a 5x5 kernel shrinks the image to nothing.
- **`Input(shape=...)`** is the first layer of the model instead of `input_shape=` in the first `Conv2D`, which current Keras versions warn about.
- **The subject only plots the validation curves.** We draw the train and the validation curves of the loss and of the accuracy, to see the overfitting.
- Wrong values (`epochs = 0`, `kernel_size = 5`, letters…) show an error message instead of crashing the page.

### Can we obtain better results?

Subject: test loss 2.8492, test accuracy 0.2864 (10 epochs, batch size 50, adam, validation_split 0.2).
Our runs use these values for everything we do not change, with 50,000 training images and the 10,000 test images.

| Run | Parameters | Test loss | Test accuracy |
|---|---|---|---|
| Subject | 270,372 | 2.8492 | 0.2864 |
| Default values of the page | 270,372 | 2.6540 | **0.3555** |
| coarse labels (20 classes) | 260,052 | 1.6908 | 0.4928 |
| filters = 64 | 679,012 | 2.5880 | 0.3624 |
| learning_rate = 0.01 | 270,372 | 4.6075 | 0.0100 |
| optimizer = sgd | 270,372 | 4.5958 | 0.0187 |

What we noticed with the default values:
- The test accuracy is 0.3555. A random guess gives 0.01 on 100 classes, so the model learned a lot, but CIFAR-100 is hard: 100 classes, small images and only 500 training images per class.
- **The model overfits.** The training accuracy keeps rising (7.6 % at epoch 1, 46.2 % at epoch 10) but the validation accuracy stops around 34.5–34.9 % from epoch 8, and the validation loss is the lowest at epoch 9 (2.6352) then goes up again (2.6661). More epochs would not help.
- We get a better score than the subject (0.2864) with the same architecture. Each number comes from one run, and the results change from one run to another because the weights start with random values.
- **Coarse labels** give the best score (0.4928), but it is an easier problem (20 classes instead of 100, a random guess gives 0.05), so this score cannot be compared with the fine labels.
- **filters = 64** gives 0.3624 against 0.3555 with 32 filters, with 2.5 times more parameters (679,012 against 270,372). The gain is small, about the size of the differences between two runs, so a bigger model is not worth it here: the problem is the overfitting, not the size of the model.
- **learning_rate = 0.01 with adam is too high**: the accuracy stays at 0.0100, the same as a random guess, and the loss (4.6075) does not go below ln(100) = 4.605, the loss of a model that learns nothing. The default 0.001 works.
- **optimizer = sgd** (learning rate 0.001) learns very slowly: the test accuracy is only 0.0187 and the loss goes from 4.6066 to 4.5962 in 10 epochs, just under ln(100). The curves still go in the right direction, so it needs a bigger learning rate or many more epochs. Unlike adam, sgd does not adapt the step of each weight.

### Role of each hyperparameter

| Hyperparameter | Role |
|---|---|
| epochs | Number of passes over the training data. Too few underfits, too many overfits (see the curves). |
| batch_size | Number of images per weight update. Small = more updates per epoch, big = faster epochs but fewer updates. |
| optimizer | How the weights are updated. adam and rmsprop adapt the step, sgd does not. |
| learning_rate | Size of the update step. Too small is slow, too big makes the training unstable. |
| validation_split | Part of the training data kept aside to watch the overfitting during the training. |
| label_mode | fine = the 100 classes, coarse = the 20 superclasses (an easier problem, scores cannot be compared with fine). |
| filters | Number of filters of the first convolution (then doubled). More filters = a bigger model that can learn more, but slower. |
| kernel_size | Size of the area seen by each filter (2x2 to 4x4). |
| dense_units | Size of the first dense layer after the convolutions. |

### Notes

- The first training downloads the dataset (about 161 MB, with a percentage on the page). The next ones start immediately.
- The training runs on the CPU and an epoch takes minutes: use 1 or 2 epochs for a quick test.
- The state of the training is kept in memory, so the application must run in a single process, and all the visitors see the same training.

### Tests

```bash
python -m unittest discover -s tests -v
```

The tests check the form values, the number of parameters (270,372 and 260,052 with the coarse labels), the state of the training (Stop, second training refused) and the `/cnn` page (form, wrong values, status, Stop). The training itself is replaced by a mock and the dataset is never downloaded.

## PW4 – PyTorch on CPU, GPU and TPU

**The report is on the site, on the page `http://localhost:5004/pw4/`** (as asked by the teacher, not in this README): what we did, what we fixed in the given notebook, the results on CPU, GPU and TPU and the conclusion.
On the same page you can upload one image and click **Inference** to compare our trained model with a pretrained model.

Files:
- `notebooks/PW4_pytorch_cpu_gpu_tpu.ipynb` – the fixed notebook (open it in Google Colab, choose CPU, GPU or TPU, then Run all)
- `notebooks/pw4_export_models.py` – trains the same CNN (2 epochs on CIFAR-10) and exports it to ONNX, and exports a MobileNetV2 pretrained on ImageNet
- `app/pw4_cifar10_cnn.onnx` (0.25 MB), `app/pw4_mobilenet_v2.onnx` (14 MB), `app/pw4_imagenet_classes.txt` – the exported models
- `app/services/pw4_inference.py`, `app/routes/pw4.py`, `app/templates/pw4_report.html` – the page

The site does not install PyTorch: the models are read with OpenCV (`cv2.dnn`).
To export them again (needs `torch` and `torchvision`):

```bash
python notebooks/pw4_export_models.py
```

Tests:

```bash
python -m unittest tests.test_pw4 -v
```

## PW5 – Land cover from a satellite image (VGG16, EuroSAT)

Notebook: `notebooks/PW5_eurosat_vgg16.ipynb` (open it in Google Colab, GPU runtime).
Page: `http://localhost:5004/landcover/`

The notebook trains a **VGG16** network on **EuroSAT** (Sentinel-2 images of 64x64 pixels, 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake) and saves the model.
The page uses this model for the **inference**: you upload a satellite screenshot (for example from Google Earth), the image is cut into square patches, every patch is classified, and the page shows the screenshot next to the land cover map, a table with the share of every class and a button to download the map as `landcover.npy`.

Files:
- `app/routes/landcover.py` – the `/landcover/` route (upload, patch size, download of the `.npy`)
- `app/services/landcover.py` – loading the model, cutting the image, classifying the patches, drawing the map
- `app/templates/landcover_dashboard.html` – the page
- `app/models_store/eurosat_vgg16.keras` and `app/models_store/class_indices.npy` – the model and the class names saved by the notebook

### The model (notebook)

- **VGG16** with the ImageNet weights, without its top layers, used as a frozen feature extractor (`pooling='avg'`), then a dense layer of 2048 neurons (relu) and a softmax layer with 10 neurons.
- Images of 64x64x3, pixels divided by 255 (`ImageDataGenerator(rescale=1./255)`), with data augmentation for the training.
- `categorical_crossentropy`, Adagrad, batch size 128, up to 50 epochs with early stopping and a checkpoint on the best validation accuracy.
- The notebook saves `vgg16_eurosat.keras` and `class_indices.npy` (the dictionary class name → number).

### The inference (the page and the notebook)

1. Load the image and keep 3 channels (RGB).
2. Loop over `y`, then over `x`: cut a square patch of `patch size` pixels.
3. Resize the patch to 64x64, divide by 255 and classify it with the model.
4. Put the class of every patch of a row in a vector, then the vectors in a matrix of the classes (one row per `y`, one column per `x`).
5. Save the matrix in `.npy` format and display it as a coloured map with a legend.

### What we paid attention to

- **The 13-band version of EuroSAT is not the right one.** `EuroSATallBands` is a 2 GB archive of `.tif` files with 13 spectral bands, but the subject asks for 3 bands (RGB). We use the RGB version: 27,000 `.jpg` images of 64x64 pixels (`EuroSAT_RGB`, downloaded by the first cell of the notebook).
- **The inference must prepare the image exactly like the training.** The model was trained with pixels divided by 255. With the VGG16 `preprocess_input` (BGR and mean subtraction) the page would give wrong classes without any error.
- **A screenshot is often an RGBA PNG** (4 channels). The image is converted to RGB, otherwise the model would get the wrong shape.
- **The scale of the image.** A EuroSAT image of 64 pixels covers about 640 m (10 m per pixel). A Google Earth screenshot has another scale, so the **patch size** can be chosen (for example 128): the patch is cut with this size and resized to 64x64 for the model.
- **The matrix has the shape `[y, x]`** (rows, columns), like every image in NumPy: its size is the number of patches in height × the number of patches in width.
- **The classes come from `class_indices.npy`**, saved by the notebook with the same order as the output of the model, and are not typed again by hand.
- The model file must be called `eurosat_vgg16.keras` in `app/models_store/` (the notebook saves it as `vgg16_eurosat.keras`, so it is renamed). If it is missing, the page shows a message instead of an error.
- `*.keras` files are ignored by git, so the model needs an exception in `.gitignore`, as for `mnist_mlp.keras`. Without it the Docker image built by the continuous integration does not contain the model.
- Elements of the screenshot that are not ground (buttons, search box, pop-up windows of Google Earth) are classified like any other patch. The screenshot should be cropped.
- The classification of 28 patches takes about 0.3 s.

### Results

Example with a Google Earth screenshot (987x622 px), patch size 128: 28 patches (4 rows × 7 columns), `landcover.npy` of shape `(4, 7)`.

| Class | Patches | Area |
|---|---|---|
| AnnualCrop | 11 | 39.3 % |
| Residential | 5 | 17.9 % |
| Highway | 4 | 14.3 % |
| PermanentCrop | 4 | 14.3 % |
| River | 2 | 7.1 % |
| HerbaceousVegetation | 1 | 3.6 % |
| Industrial | 1 | 3.6 % |

<!-- Paste here the test accuracy and the confusion matrix printed by the notebook -->

What we noticed:
- The screenshot is mostly farmland, so `AnnualCrop` and `PermanentCrop` dominate, and `Residential` appears where the villages are. The result looks plausible.
- Some patches are wrong (for example `Highway` on field roads and parcel limits). The model learned on Sentinel-2 images (colours, resolution and angle of the sensor), and a Google Earth image looks different: the score on the EuroSAT test set is therefore better than what we see on a screenshot.
- The notebook trains on 20 % of the images only (`train_size = 0.2`, to keep the training short on a free Colab GPU), so the model could be improved with more data.
- The result changes with the patch size, because the patch must contain enough of one kind of ground: too small, it sees only a part of a field, too big, it mixes several kinds of ground.

## Object detection with YOLO

Page: `http://localhost:5004/yolo/`

You upload a photo and a small ready model, **YOLOv8n** (trained on the 80 classes of the COCO dataset: person, car, bus, dog, cup...), draws a box around every object with its name and its confidence. The confidence threshold can be changed in the form. The photo is not saved on the server.

Files:
- `app/yolov8n.onnx` – the model (12.9 MB)
- `app/services/yolo.py` – preparing the photo, running the model, reading the result, drawing the boxes
- `app/routes/yolo.py` – the `/yolo/` route
- `app/templates/yolo_dashboard.html` – the page

### Why ONNX and OpenCV instead of the `ultralytics` package

The easiest way to use YOLO is the `ultralytics` package, but it installs **PyTorch**, which would make our Docker image (already about 4.3 GB because of TensorFlow) much bigger. So we exported the model once to the **ONNX** format (in a separate container, with `yolo export model=yolov8n.pt format=onnx`) and we read it with **OpenCV** (`cv2.dnn`), which is already in the requirements. The site does not need PyTorch at all.

We measured the size in a clean `python:3.12-slim` container:

| What is installed | Size |
|---|---|
| `opencv-python-headless` (what we use) | 223 MB |
| `torch` + `torchvision` (CPU) + `ultralytics` on top | 958 MB more |
| the model `yolov8n.onnx` | 12 MB |

So the ONNX + OpenCV way saves almost **1 GB** in the Docker image (and the CUDA version of PyTorch, installed by default, is even bigger).

What we had to do ourselves, because `ultralytics` normally does it:
- **Letterbox**: the model wants a 640×640 image, so the photo is resized without changing its proportions and grey bands are added.
- **Reading the output**: the model returns a table of 84 × 8400 (4 numbers for the box + 80 class scores, for 8400 positions). We keep the positions above the threshold and convert the boxes back to the size of the original photo.
- **NMS** (non-maximum suppression, `cv2.dnn.NMSBoxes`): the model finds the same object several times, NMS keeps only the best box.
- Photos from a phone can be rotated (EXIF), so they are turned the right way first.

Problem we met: `opencv-python` does not import in the `python:3.12-slim` image (`libxcb.so.1: cannot open shared object file`), because it needs graphical libraries. We use **`opencv-python-headless`** (same version), which is made for servers.

Result on the standard test photo of Ultralytics (`bus.jpg`): 1 bus and 4 persons found (83–90 %), in less than a second on the CPU.

### Tests

```bash
python -m unittest tests.test_yolo -v
```

The tests that run the model need OpenCV (they are skipped without it).

## Continuous integration (Session 7)

File: `.github/workflows/ci.yml` (GitHub Actions).

On every push and every pull request, GitHub runs two jobs:
1. **Unit tests** – installs Python 3.12 (the same version as the Dockerfile) and `requirements.txt`, then runs `python -m unittest discover -s tests -v`.
2. **Docker image** – builds the image exactly like `docker compose` (`docker build -f app/Dockerfile .`), starts it and checks that the home page answers.

A pull request should only be merged when both jobs have a green check mark on GitHub (tab **Actions**, or at the bottom of the pull request).

Notes:
- The tests use the real `chinook.db` file and the saved models, so no database server is needed in CI. InfluxDB is not reachable from GitHub, so the InfluxDB pages are not tested against the real server.
- PostgreSQL is not started in CI either: without `DATABASE_URL`, our own tables (the digit models) are created in a local SQLite file, and the two given models are copied into it at the start.
- The image is about 4.3 GB because of TensorFlow, so the Docker job takes a few minutes.

## Security
- Do not commit `.env` files or API tokens.
- Keep credentials out of source code. The default PostgreSQL password (`password123`) is only for local work: set `POSTGRES_PASSWORD` in `.env` for any shared server.
- Do not expose Flask debug mode to untrusted networks.
- Importing a Keras model from a user can run code (especially `.h5`): only import models you trust.

## Academic Context

This project demonstrates Flask web development, database integration (SQLite and PostgreSQL), InfluxDB time-series queries and forecasting, machine learning and deep learning models (classical ML, MLP, CNN, LSTM, object detection), storing and serving models from a database, data visualisation and continuous integration.