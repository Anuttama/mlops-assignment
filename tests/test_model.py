import os
import sys

# Adjust the Python path to include the root directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from model.model import (  # noqa: E402
    load_and_preprocess_data,
    train_test_split_data,
    train_and_evaluate_model,
)


def test_load_and_preprocess_data():
    dataset_path = "data/dataset.csv"
    df = load_and_preprocess_data(dataset_path)
    assert df is not None
    assert "age" in df.columns


def test_train_test_split_data():
    dataset_path = "data/dataset.csv"
    df = load_and_preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0


def test_train_and_evaluate_model():
    dataset_path = "data/dataset.csv"
    df = load_and_preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    accuracy, logreg = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    assert 0 <= accuracy <= 1
