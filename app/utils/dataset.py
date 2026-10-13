
def format_dataset(data, x):
    X = []
    y = []

    for i in range(len(data) - 2 * x + 1):
        X.append(data[i:i + x])
        y.append(data[i + x:i + 2 * x])

    return X, y