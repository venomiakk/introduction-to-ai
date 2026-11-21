import csv
from pathlib import Path
import numpy as np


IRIS_FILE = Path(__file__).resolve().parent / "iris.csv"


def read_iris_csv(filepath=IRIS_FILE):
    """Reads the local Iris CSV file.

    Returns:
        X: np.ndarray of shape (150, 4) with float64 feature values.
        y: np.ndarray of shape (150,) with integer class indices (0, 1, 2).
        feature_names: list[str] of the 4 feature column names.
        class_names: list[str] of the 3 sorted Iris species names.
    """
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[:-1]

        rows_X = []
        raw_labels = []
        for row in reader:
            if not row:
                continue
            rows_X.append([float(val) for val in row[:-1]])
            raw_labels.append(row[-1])

    X = np.array(rows_X, dtype=np.float64)
    class_names = sorted(set(raw_labels))
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.array([label_to_idx[label] for label in raw_labels], dtype=np.int64)

    return X, y, feature_names, class_names


def one_hot_encode(y, num_classes):
    """Converts integer class indices of shape (N,) into one-hot matrix (N, num_classes)."""
    one_hot = np.zeros((len(y), num_classes), dtype=np.float64)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    """Splits (X, y) into train and test sets while preserving class proportions."""
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_test = max(1, int(round(len(cls_indices) * test_size)))
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_features(X_train, X_test):
    """Standardizes features to zero mean and unit variance using only training statistics."""
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled, mean, std


def load_iris_data(test_size=0.2, seed=42, standardize=True):
    """Loads, splits, normalizes, and one-hot encodes the Iris dataset."""
    X, y, feature_names, class_names = read_iris_csv()
    X_train_raw, X_test_raw, y_train, y_test = stratified_train_test_split(
        X, y, test_size=test_size, seed=seed
    )

    if standardize:
        X_train, X_test, mean, std = standardize_features(X_train_raw, X_test_raw)
    else:
        X_train, X_test = X_train_raw.copy(), X_test_raw.copy()
        mean = np.zeros((1, X.shape[1]), dtype=np.float64)
        std = np.ones((1, X.shape[1]), dtype=np.float64)

    num_classes = len(class_names)
    Y_train_onehot = one_hot_encode(y_train, num_classes)
    Y_test_onehot = one_hot_encode(y_test, num_classes)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "Y_train_onehot": Y_train_onehot,
        "Y_test_onehot": Y_test_onehot,
        "feature_names": feature_names,
        "class_names": class_names,
        "mean": mean,
        "std": std,
    }
