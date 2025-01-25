import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_splits
import os
from sklearn.metrics import accuracy_score


def load_and_preprocess_data(dataset_path):
    # Load and preprocess dataset
    disease_df = pd.read_csv(dataset_path)
    disease_df.drop(['education'], inplace=True, axis=1)
    disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
    disease_df.dropna(axis=0, inplace=True)
    return disease_df

def train_test_split_data(disease_df):
    # Prepare features and labels
    X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 
                               'totChol', 'sysBP', 'glucose']])
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
    return accuracy

if __name__ == "__main__":
    dataset_path = "data/dataset.csv"
    disease_df = load_and_preprocess_data(dataset_path)
    print(disease_df.head(), disease_df.shape)
    print(disease_df.TenYearCHD.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split_data(disease_df)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)
    
    accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print('Accuracy of the model is =', accuracy)
