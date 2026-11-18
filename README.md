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

### CIFAR-100 CNN (PW3)
- Train a convolutional neural network on CIFAR-100 (the dataset is imported by the page).
- Change the hyperparameters from the web page (fine or coarse labels, filters, kernel size, optimizer...).
- Follow the download and the training live, and stop them with a button.

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
| coarse labels (20 classes) | 260,052 | TODO | TODO |
| filters = 64 | 679,012 | TODO | TODO |
| dense_units = 512 | 512,932 | TODO | TODO |
| learning_rate = 0.0005 | 270,372 | TODO | TODO |
| learning_rate = 0.01 | 270,372 | TODO | TODO |
| optimizer = rmsprop | 270,372 | TODO | TODO |
| optimizer = sgd | 270,372 | TODO | TODO |
| batch_size = 128 | 270,372 | TODO | TODO |

What we noticed with the default values:
- The test accuracy is 0.3555. A random guess gives 0.01 on 100 classes, so the model learned a lot, but CIFAR-100 is hard: 100 classes, small images and only 500 training images per class.
- **The model overfits.** The training accuracy keeps rising (7.6 % at epoch 1, 46.2 % at epoch 10) but the validation accuracy stops around 34.5–34.9 % from epoch 8, and the validation loss is the lowest at epoch 9 (2.6352) then goes up again (2.6661). More epochs would not help.
- We get a better score than the subject (0.2864) with the same architecture. Each number comes from one run, and the results change from one run to another because the weights start with random values.
- TODO: comment the table above (coarse labels, filters, learning rate, optimizer, batch size).

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

## Security
- Do not commit `.env` files or API tokens.
- Keep credentials out of source code.
- Do not expose Flask debug mode to untrusted networks.

## Academic Context

This project demonstrates Flask web development, database integration, InfluxDB time-series queries, data visualisation, and dataset formatting.