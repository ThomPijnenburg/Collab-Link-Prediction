from typing import Callable

from sklearn.ensemble import RandomForestClassifier

from .data import Dataset


def create_split_X_Y() -> Callable:
    def split_X_Y(dataset: Dataset) -> tuple:
        X = dataset.features
        Y = dataset.labels
        return (X, Y)
    return split_X_Y


def create_train_RandomForestClassifier(nb_tree):
    rf = RandomForestClassifier(n_estimators=nb_tree)
    def train_rf_on(training_set):
        X, Y = training_set
        rf.fit(X,Y)
        def predict_rf_on(external_dataset):
            return rf.predict(external_dataset)
        return predict_rf_on
    return train_rf_on


