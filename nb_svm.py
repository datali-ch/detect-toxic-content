from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy.sparse import csr_matrix
from config import LOG_REGRESSION_SOLVER
from pandas import DataFrame
from numpy import ndarray
from typing import List, Tuple, Union

FEATURES_FORMAT = Union[DataFrame, ndarray]


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    # Navie Bayes - Support Vector Machines Classifier
    # This implementation uses Logistic Regression instead of SVM, which produces nearly identical outcomes
    #
    # Credits to: Alex Sanchez
    # https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb#261316

    def __init__(
        self, C: float = 1.0, n_jobs: int = -1, solver: str = LOG_REGRESSION_SOLVER
    ) -> None:
        """ Initialize NB-SVM Classifier

            Args: 
                C:          inverse of regularization strength
                n_jobs:     number of CPU cores used when parallelizing over classes. -1 for all processors     
                solver:     algorithm to use in the optimization problem. Allowed values: ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}

            Returns:
                None
        """

        self.C = C
        self.n_jobs = n_jobs
        self.solver = solver

    def predict(self, x: ndarray) -> ndarray:
        """ Given fitted model, predicts label [0,1] for a set of features
            
            Args:
                x (N,M):          M features over N examples

            Returns:
                _ (N,1):          predicted labels over N examples
        """

        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict(x.multiply(self._r))


    def predict_proba(self, x: ndarray) -> ndarray:
        """ Given fitted model, predicts probability of label=1 for a set of features
            
            Args:
                x (N,M):          M features over N examples

            Returns:
                _ (N,1):          probability of label=1 over N examples
        """

        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict_proba(x.multiply(self._r))


    def fit(self, x: ndarray, y: ndarray) -> None:
        """ Given set of features, fits logistic regression classifier
            
            Args:
                x (N,M):          M features over N examples
                y (N,1):          labels [0,1] over N examples

            Returns:
                None
        """

        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def probability(x: ndarray, y_i: int, y: ndarray) -> float:
            """ Given set of features and labels, calculates probability of given label
                
                Args:
                    x (N,M):          M features over N examples
                    y_i:              label for which probability is returned
                    y (N,1):          labels [0,1] over N examples

                Returns:
                    None
            """

            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = csr_matrix(np.log(probability(x, 1, y) / probability(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(
            C=self.C, solver=self.solver, n_jobs=self.n_jobs
        ).fit(x_nb, y)
        return self


def fitModel(
    model: NbSvmClassifier,
    X_train: FEATURES_FORMAT,
    X_test: FEATURES_FORMAT,
    y_train: DataFrame,
    y_test: DataFrame,
    labels: List[str],
) -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """ Fits multiclass classification model using one-vs-all method

        Args:
            model:                             Model for binary classification
            X_train (N,M):                     features for training set. N examples, M features
            X_test  (K,L):                     features for test set. K examples, L features
            y_train (N,P):                     labels for training_set (one-hot encoding: N examples, P labels)
            y_test  (K,P):                     labels for test_set     (one-hot encoding: K examples, P labels)
            labels  (1,P):                     names of P features 

        Returns:
            metrics_train (4, P):              Accuracy, F1 score, Precision and Recall for predicitions on training set
            metrics_test  (4, P):              Accuracy, F1 score, Precision and Recall for predicitions on test set
            MEASURES      (1, 4):              names of metrics returned
    """

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

        for j, function in enumerate(FUNCTIONS):

            metrics_train[j][i] = function(model.predict(X_train), y_train[label])
            metrics_test[j][i] = function(model.predict(X_test), y_test[label])

    return metrics_train, metrics_test, MEASURES
