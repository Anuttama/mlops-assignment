import pandas as pd
import numpy as np
from sklearn import preprocessing
'exec(% matplotlib inline)'
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import sys
# Adjust the Python path to include the root directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)


def load_and_preprocess_data(dataset_path):
    # Load and preprocess dataset
    disease_df = pd.read_csv(dataset_path)
    disease_df.drop(['education'], inplace=True, axis=1)
    disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
    disease_df.dropna(axis=0, inplace=True)
    return disease_df


def train_test_split_data(disease_df):
    # Prepare features and labels
    X = np.asarray(
        disease_df[
            [
                'age',
                'Sex_male',
                'cigsPerDay',
                'totChol',
                'sysBP',
                'glucose',
            ]
        ]

    )
    y = np.asarray(disease_df['TenYearCHD'])
    # Normalize features
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # Train-test split
    return train_test_split(X, y, test_size=0.3, random_state=4)


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Train logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, logreg


if __name__ == "__main__":
    dataset_path = "dataset.csv"
    disease_df = load_and_preprocess_data(dataset_path)
    print(disease_df.head(), disease_df.shape)
    print(disease_df.TenYearCHD.value_counts())
    X_train, X_test, y_train, y_test = train_test_split_data(disease_df)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)
    accuracy, logreg = train_and_evaluate_model(
                        X_train, X_test, y_train, y_test)
    print('Accuracy of the model is =', accuracy)
    # Save the trained model as a pickle file
    with open("model.pkl", "wb") as file:
        pickle.dump(logreg, file)
