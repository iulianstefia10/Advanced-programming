# Advanced Programming

A Flask web application developed for the R5.A.05 Advanced Programming practical exercises at IUT Annecy.

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
- Compare the prediction of the given CNN and of our PW2 MLP, with the probabilities.

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

## Technologies
- Python 3.12, Flask, Jinja2
- Flask-SQLAlchemy and SQLite (Chinook database)
- InfluxDB 2.x and the Flux language (time series)
- scikit-learn (random forest, K-Means, linear regression)
- TensorFlow / Keras (MLP, CNN, LSTM)
- OpenCV and ONNX (YOLO object detection)
- NumPy and pandas
- PyTorch in Google Colab (PW4)
- HTML, CSS (Bootstrap) and JavaScript (charts drawn in SVG)
- Docker and Docker Compose
- GitHub Actions (continuous integration)

## Project Structure

```text
Advanced-Programming/
├── .github/workflows/ci.yml      # tests + Docker build on every push (Session 7)
├── app/
│   ├── app.py                    # creates the Flask app, registers every page
│   ├── extensions.py             # SQLAlchemy object
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── database/chinook.db       # SQLite database (Sessions 2-3, Ex 1)
│   ├── models/                   # SQLAlchemy models (Chinook, inference models)
│   ├── routes/                   # one blueprint per page
│   │   ├── chinook.py            # /chinook/...          Sessions 2-3, Ex 1 + clusters
│   │   ├── influxdb.py           # /influxdb/...         Sessions 2-3, Ex 2
│   │   ├── forecast.py           # /forecast/            time-series forecasting
│   │   ├── apparent.py           # /apparent/            Session 1, 5.4
│   │   ├── mnist.py              # /mnist/               Sessions 2-3, Ex 3
│   │   ├── random_forest.py      # /rf/                  PW1
│   │   ├── mlp.py                # /mlp/                 PW2
│   │   ├── cnn.py                # /cnn/                 PW3
│   │   └── yolo.py               # /yolo/                object detection
│   ├── services/                 # the models and the computations of each page
│   ├── utils/dataset.py          # windows for time series (format_dataset, make_windows)
│   ├── templates/                # one page per feature + base.html and home.html
│   ├── static/                   # CSS, uploaded images (not committed)
│   ├── MNIST_keras_CNN.h5        # given CNN (Ex 3)
│   ├── mnist_mlp.keras           # our PW2 MLP, saved by train_mnist_mlp.py
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
- the number of invoices,
- the number of different genres bought,
- the favourite genre and its share of the tracks bought (%).

The four numbers are scaled with `StandardScaler` (they have different units) and grouped with K-Means.
`k` is chosen in the form (2 to 10). The page shows the elbow curve, a scatter plot coloured by cluster and the table of the clusters and of the customers.

Files:
- `app/services/chinook_clusters.py` – queries, K-Means, elbow curve and plots
- `app/routes/chinook.py` – the `/chinook/clusters` route
- `app/templates/chinook_clusters.html` – the page

### What we noticed

- The elbow is around **k = 4**: the inertia goes 236 → 162 → 101 → 64, then decreases slowly.
- With k = 4: a group of 8 big spenders who buy many genres (45.6 $, 9.5 genres), a group of 23 customers who mostly buy their favourite genre (51 % of their tracks), and a group of 27 "average" customers.
- Chinook is a demo database: 58 of the 59 customers have exactly 7 invoices. So the number of invoices hardly separates the customers. The only customer with 6 invoices always gets a cluster of its own after scaling.
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
3. **Windows**: the last x hours of all the series are the input, the next hours of the target are the output (`make_windows`; with one variable it gives the same windows as `format_dataset`).
4. **Split by time, never shuffled**: the first 80 % of the windows to train, the last 20 % to test, with a gap between them so that no hour is used both for training and for testing. The scaling (mean and standard deviation) is computed on the training hours only.
5. **Five models**:

| Model | Idea |
|---|---|
| Naive | the next hours equal the last known value |
| Seasonal naive | the next hours equal the same hours of the previous day (temperature has a 24 h cycle) |
| Linear regression | ridge regression on all the values of the window |
| MLP | dense network 64 → 32 → horizon (TensorFlow) |
| LSTM | recurrent network with 32 units that reads the window hour by hour (TensorFlow) |

The MLP and the LSTM stop when the error on the most recent 10 % of the training windows stops improving (early stopping, patience 5).

6. **Errors** on the test period, in °C: MAE (mean absolute error), RMSE, and the MAE for every forecast hour (+1 h … +x h). Every learned model is trained twice, **with all the inputs** and **with the Annecy temperature only**; the column "gain" shows how much the other variables and cities help.

### What we paid attention to

- **No shuffling and a gap between train and test.** With overlapping windows, a random split puts almost the same hours in train and test and gives errors that are far too good.
- **Scaling on the training part only**, for the same reason.
- **Baselines first.** Temperature has a strong daily cycle; a model is only useful if it beats "same hour yesterday".
- The teacher's query reads **one week** (`range(start: -1w)`); that is about 170 hours, too few windows for an LSTM. The page reads 30 days by default (up to 365).
- `crypto` and `forex` have other tags and fields: the page reads them from the bucket, so the same code works on them.

### Results

<!-- Paste here the output of: docker exec group4 python forecast_report.py -->

### Tests

```bash
python -m unittest tests.test_forecast_service tests.test_forecast_route -v
```

The tests use a fake InfluxDB (two stations, one of them sees the weather 6 hours earlier) so they run without the server, also on GitHub. They check the windows, that train and test never share an hour, the Flux query, the form checks, that the linear model beats the naive baseline and that the other city helps. The MLP and LSTM test is skipped when TensorFlow is not installed.

## Session 3 – Digit recognition: draw a digit, CNN and MLP

Page: `http://localhost:5004/mnist/`

You can draw a digit with the mouse (or a finger on a phone) or upload an image.
The same prepared image is sent to two models and both predictions are shown with the probability of each digit:
- **CNN**: `MNIST_keras_CNN.h5`, the model given in the subject,
- **MLP**: our PW2 network (128-64-32, pixels divided by 255), trained once with `app/train_mnist_mlp.py` and saved in `app/mnist_mlp.keras` (test accuracy 0.974).

Files:
- `app/services/mnist_inference.py` – preparing the image and running the two models
- `app/routes/mnist.py` – the `/mnist/` route (drawing or upload)
- `app/templates/mnist_dashboard.html` – the canvas and the results
- `app/train_mnist_mlp.py` – trains and saves the MLP (run it again only if the model must change)

### What we paid attention to

- The two models do not take the same input: the CNN wants an image `(1, 28, 28, 1)`, the MLP wants a vector of **784** values. Both must be divided by **255**, like during the training.
- MNIST digits are **white on a black background**, but a drawing is black on white, so the image is inverted. A transparent background (canvas, PNG) is first made white.
- In MNIST the digit is about 20×20 pixels and centred in 28×28. Before, the whole picture was just resized, so a digit drawn in a corner or very big was badly recognised (a 7 was predicted as 1). Now the digit is cropped, fitted in a 20×20 box and moved so that its centre of mass is in the middle. The page shows this 28×28 image ("what the models see").
- With this preparation, on 200 MNIST test images turned into big black-on-white drawings, the CNN is right 199 times and the MLP 195 times. The CNN is a bit better on unusual shapes (for example a 6 written with a font).
- An uploaded file is saved in `static/uploads` only if it really is an image.
- `*.keras` files are ignored by git (PW3 models), so `app/mnist_mlp.keras` is added as an exception in `.gitignore`.

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

Notebook: `notebooks/PW4_pytorch_cpu_gpu_tpu.ipynb` (open it in Google Colab).

A small CNN is trained on CIFAR-10 for 2 epochs, then tested on the 10,000 test images.
To compare the hardware: **Runtime > Change runtime type** (CPU, GPU or TPU), then **Run all**. The last cell prints the times.

### What we fixed in the given notebook

- **Errors 1, 2 and 3 (`Input type`)**: the model was on the GPU but the test images stayed on the CPU. In the first one, the code moved `inputs` (from the training loop) instead of `images`. We send `images` and `labels` to the device in the three test cells.
- The **inference time** was not measured, we added it.
- The GPU and TPU work asynchronously, so we synchronize before reading the time.
- **TPU**: the notebook only looked for `torch_xla`, which is not installed on the new Colab TPU runtime, so it silently ran on the CPU. The new runtime uses `torch_tpu`, so we choose the device with `torch.accelerator` (GPU or TPU), and keep `torch_xla` for older runtimes.
- On the TPU runtime the `DataLoader` worker processes crashed the session, so we use `num_workers=0` there.

### Results

| Hardware | Training (2 epochs) | Inference (10,000 images) | Accuracy |
|---|---|---|---|
| CPU (our laptop) | 113 s | 9 s | 55 % |
| Colab CPU (Intel Xeon 2.2 GHz) | 168 s | 10 s | 55 % |
| Colab GPU (T4) | 119 s | 7 s | 54 % |
| Colab TPU (v5e-1) | 96 s | 8 s | 55 % |

### Conclusion

- The TPU was the fastest for training and the GPU for inference, but the differences are small: the GPU is only 1.4 times faster than the Colab CPU.
- The reason is the batch size of 4 and the very small model: each step sends only 4 images to the accelerator and does very little computation, so most of the time is spent on transfers and loading the data, not on computing. With bigger batches and a bigger model, the GPU and TPU would be much faster.
- Our laptop CPU was faster than the Colab CPU, so for such a small model a GPU is not really needed.
- Using the TPU with PyTorch is not automatic: without the right library the code runs on the CPU without any error.

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
- The image is about 4.3 GB because of TensorFlow, so the Docker job takes a few minutes.

## Security
- Do not commit `.env` files or API tokens.
- Keep credentials out of source code.
- Do not expose Flask debug mode to untrusted networks.

## Academic Context

This project demonstrates Flask web development, database integration, InfluxDB time-series queries and forecasting, machine learning and deep learning models (classical ML, MLP, CNN, LSTM, object detection), data visualisation and continuous integration.