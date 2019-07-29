# -*- coding: utf-8 -*-

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
import scipy
from pandas import DataFrame
from config import (
    LOG_REGRESSION_SOLVER,
    STOP_WORDS,
    STRIP_ACCENTS,
    MAX_FEATURES,
    MIN_DF,
    C,
)
from typing import List, Tuple

def trainBagOfWords(X_train, X_test, y_train, y_test, labels):

    model = NbSvmClassifier(C=C)
    metrics_train, metrics_test, measures = fitModel(
        model, X_train, X_test, y_train, y_test, labels
    )
    return metrics_train, metrics_test, measures


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


def dummy_fun(doc):
    return doc

def dummy_toc(doc):
    return doc.split()

def calculateTFIDFscore(df: DataFrame, ngrams: Tuple[int, int]=(1,1)) -> Tuple[scipy.sparse.csr_matrix, List[str]]:
    """ Calculates Term Frequency - Inverse Document Frequency Score 

        Args:
            df:                               texts to score, one sample per row
            ngrams:                           range of n-grams to include: (from, to). Default: unigrams   

        Returns:
            word_counts:                      term-document matrix [n_samples, n_features]
            features:                         features corresponding to cols of word_counts
    """
    
    tfv = TfidfVectorizer(
        ngram_range=ngrams,
        lowercase=False,
        stop_words=STOP_WORDS,
        strip_accents=STRIP_ACCENTS,
        preprocessor = dummy_fun,
        tokenizer = dummy_toc,
        analyzer="word",
        max_features=MAX_FEATURES,
        min_df=MIN_DF,      
    )

    word_counts = tfv.fit_transform(df)
    features = np.array(tfv.get_feature_names())

    return word_counts, features


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    # Credits to: Alex Sanchez
    # https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb#261316

    def __init__(self, C=1.0, n_jobs=-1, solver="lbfgs"):
        self.C = C
        self.n_jobs = n_jobs
        self.solver = solver

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

        def probability(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = scipy.sparse.csr_matrix(np.log(probability(x, 1, y) / probability(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(
            C=self.C, solver=self.solver, n_jobs=self.n_jobs
        ).fit(x_nb, y)
        return self
