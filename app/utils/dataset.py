
def format_dataset(data, x):
    X = []
    y = []

    for i in range(len(data) - 2 * x + 1):
        X.append(data[i:i + x])
        y.append(data[i + x:i + 2 * x])

    return X, y

def make_windows(values, target_col, window, horizon):
    """Multivariate version of format_dataset (Sessions 2-3, 7.4).

    values: 2-D array (time, variables), one row per hour.
    X[i] = all variables of the `window` rows starting at i,
    y[i] = the target variable of the `horizon` rows that follow.
    With one variable and window == horizon == x it gives the same
    windows as format_dataset(data, x).
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    n = len(values) - window - horizon + 1
    if n <= 0:
        return (np.empty((0, window, values.shape[1])), np.empty((0, horizon)))
    X = np.stack([values[i:i + window] for i in range(n)])
    y = np.stack([values[i + window:i + window + horizon, target_col] for i in range(n)])
    return X, y
