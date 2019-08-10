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
from config import LOG_REGRESSION_SOLVER, MAX_ITER
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
                output (N,K):     predicted labels over N examples and K classes (one hot encoding)
        """

        check_is_fitted(self, ["_r", "_clf"])
        x = csr_matrix(x)
        output = np.zeros((x.shape[0], len(self._clf)))

        for i in range(len(self._clf)):
            output[:,i] = self._clf[i].predict(x.multiply(self._r[i,:]))

        return output

    def predict_proba(self, x: ndarray) -> ndarray:
        """ Given fitted model, predicts probability of label=1 for a set of features

            Args:
                x (N,M):          M features over N examples

            Returns:
                output (N,K):     probability of label=1 over N examples and K classes
        """

        check_is_fitted(self, ["_r", "_clf"])
        x = csr_matrix(x)
        output = np.zeros((x.shape[0], len(self._clf)))

        for i in range(len(self._clf)):
            output[:,i] = self._clf[i].predict_proba(x.multiply(self._r[i,:]))

        return output


    def fit(self, x: ndarray, y: ndarray) -> None:
        """ Given set of features, fits logistic regression classifier

            Args:
                x (N,M):          M features over N examples
                y (N,K):          labels for K classes, one-hot encoding

            Returns:
                self
        """

        x, y = check_X_y(x, y, accept_sparse=True, multi_output=True)

        def probability(x: ndarray, y_i: int, y: ndarray) -> float:
            """ Given set of features and labels, calculates probability of given label

                Args:
                    x (N,M):                M features over N examples
                    y_i:                    label for which probability is returned
                    y (N,K):                labels for K classes, one-hot encoding

                Returns:
                    probabilities (M,K)     probabilities of label y_i
            """

            probabilities = np.zeros((y.shape[1], x.shape[1]))
            for i in range(y.shape[1]):
                ind2take = y[:,i] == y_i
                probabilities[i,:] = (x[ind2take].sum(0) + 1) / (ind2take.sum() + 1)

            return probabilities

        self._r = csr_matrix(np.log(probability(x, 1, y) / probability(x, 0, y)))
        self._clf = [0]*y.shape[1]
        classifier = LogisticRegression(C=self.C, solver=self.solver, n_jobs=self.n_jobs, max_iter=MAX_ITER)
        x = csr_matrix(x)

        for i in range(y.shape[1]):

            x_nb = x.multiply(self._r[i,:])
            self._clf[i] = classifier.fit(x_nb, y[:,i])

        return self


def calculateModelMetrics(y_pred: ndarray, y_true: ndarray) -> List[List[float]]:
    """ Calculates accuracy, F1 score, precision and recall for multi-label classification

        Args:
            y_pred  (N,P):                     predicted labels (one-hot encoding: N examples, P labels)
            y_true  (N,P):                     true labels (one-hot encoding: N examples, P labels)

        Returns:
            metrics       (4, P):              Accuracy, F1 score, Precision and Recall for predicitions
            MEASURES      (1, 4):              names of metrics returned
    """

    MEASURES = ["Accuracy", "F1 score", "Precision", "Recall"]
    FUNCTIONS = [accuracy_score, f1_score, precision_score, recall_score]

    num_classes = y_pred.shape[1]
    metrics = [[0] * num_classes for i in range(len(MEASURES))]

    for i, function in enumerate(FUNCTIONS):
        for j in range(num_classes):
            metrics[i][j] = function(y_pred[:,j], y_true[:,j])

    return metrics, MEASURES
