# Train the PW2 MLP once and save it for the /mnist/ page
# Run from the app folder: python train_mnist_mlp.py

import os

import tensorflow as tf

from services.mlp import SEED, build_model, load_mnist

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_mlp.keras")


def main():
    tf.keras.utils.set_random_seed(SEED)

    # same model as in PW2: 128-64-32, pixels divided by 255
    X_train, y_train, X_test, y_test = load_mnist(normalize=True)
    model = build_model([128, 64, 32], activation="relu")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.1, verbose=2)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f} / Test accuracy: {test_acc:.4f}")

    model.save(MODEL_PATH)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
