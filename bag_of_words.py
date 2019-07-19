# -*- coding: utf-8 -*-

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy import sparse
from config import LOG_REGRESSION_SOLVER


def fitModel(model, X_train, X_test, y_train, y_test, labels):

    MEASURES = ["Accuracy", "F1 score", "Precision", "Recall"]
    FUNCTIONS = [accuracy_score, f1_score, precision_score, recall_score]

    pred_train = np.zeros((X_train.shape[0], len(labels)))
    pred_test = np.zeros((X_test.shape[0], len(labels)))

    metrics_train = [[0] * len(labels) for i in range(len(MEASURES))]
    metrics_test = [[0] * len(labels) for i in range(len(MEASURES))]

    # Fits one variable at a time
    for i, label in enumerate(labels):

        model.fit(X_train, y_train[label])
        pred_test[:, i] = model.predict_proba(X_test)[:, 1]
        pred_train[:, i] = model.predict_proba(X_train)[:, 1]

        loss = log_loss(y_train[label], pred_train[:, i])
        print("Class: " + label)
        print("Log loss:", loss)

        for j, function in enumerate(FUNCTIONS):

            metrics_train[j][i] = function(model.predict(X_train), y_train[label])
            metrics_test[j][i] = function(model.predict(X_test), y_test[label])

    return metrics_train, metrics_test, MEASURES


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(
            C=self.C, solver = 'lbfgs', max_iter=10000
        ).fit(x_nb, y)
        return self
