import os

import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="module")
def data():
    """Load the census data once for all tests."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def split_data(data):
    """Split the data the same way train_model.py does."""
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )
    return train, test


@pytest.fixture(scope="module")
def processed(split_data):
    """Process a small sample of the data so the tests run quickly."""
    train, test = split_data
    train = train.sample(2000, random_state=42)
    test = test.sample(500, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb, test


@pytest.fixture(scope="module")
def model(processed):
    """Train a model on the small sample."""
    X_train, y_train = processed[0], processed[1]
    return train_model(X_train, y_train)

# TODO: implement the first test. Change the function name and input as needed
def test_data_shape_and_columns(data):
    """Check that the data has the expected columns, no nulls, and two salary labels."""
    assert data.shape[0] > 30000
    assert data.shape[1] == 15
    for col in CAT_FEATURES + ["salary"]:
        assert col in data.columns
    assert data.isnull().sum().sum() == 0
    assert set(data["salary"].unique()) == {"<=50K", ">50K"}


# TODO: implement the second test. Change the function name and input as needed
def test_train_test_split_size(data, split_data):
    """Check that the train and test datasets are DataFrames with an 80/20 split."""
    train, test = split_data
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(0.20 * len(data), abs=1)


# TODO: implement the third test. Change the function name and input as needed
def test_train_model_algorithm(model):
    """Check that train_model returns a fitted GradientBoostingClassifier."""
    assert isinstance(model, GradientBoostingClassifier)
    assert hasattr(model, "estimators_")


def test_inference_output(model, processed):
    """Check that inference returns one 0/1 prediction per row."""
    X_test = processed[2]
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_values():
    """Check that compute_model_metrics returns the expected values on a known example."""
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 1, 1])
    # TP = 2, FP = 1, FN = 1
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)
