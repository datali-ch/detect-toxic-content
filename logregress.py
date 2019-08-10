import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from nb_svm import NbSvmClassifier
from config import TFIDF_FILE, LABELS, DATA_FILE, C, CONTENT
from features_engineering import loadSparseMatrix, calculateTFIDFscore
from nb_svm import NbSvmClassifier, NbSvmMultilabel, calculateModelMetrics
from load_data import loadData
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

df = loadData(DATA_FILE)
features, _ = calculateTFIDFscore(df[CONTENT])

X_train, X_test, y_train, y_test = train_test_split(features, df[LABELS], test_size=TEST_SIZE, random_state=123)

# Old version

model = NbSvmClassifier(C=C)
models_old = [[0] for i in range(len(LABELS))]


# Fits one variable at a time
for i, label in enumerate(LABELS):

    model.fit(csr_matrix.toarray(X_train), y_train[label])
    models_old[i] = model._clf


model = NbSvmMultilabel(C=C)
model.fit(csr_matrix.toarray(X_train), y_train.values)
models_new = model._clf


for model_old, model_new in zip(models_old, models_new):
    A = (model_old.coef_==model_new.coef_).all()
    B = (model_old.intercept_==model_new.intercept_).all()
    C = model_old.classes_==model_new.classes_
    D = (model_old.n_iter_==model_new.n_iter_).all()
    print(A and B)
