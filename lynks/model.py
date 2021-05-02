from sklearn.ensemble import RandomForestClassifier


def create_train_RandomForestClassifier(nb_tree):
    rf = RandomForestClassifier(n_estimators=nb_tree)

    def train_rf_on(X, Y):
        rf.fit(X, Y)

        def predict_rf_on(X):
            return rf.predict(X)
        return predict_rf_on
    return train_rf_on
