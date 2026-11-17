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
