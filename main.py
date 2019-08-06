# -*- coding: utf-8 -*-

import argparse
from sklearn.model_selection import train_test_split
import os.path
from load_data import loadData
import pandas as pd
from config import (
    DATA_FILE,
    TEST_SIZE,
    CONTENT,
    LABELS,
    C,
    STOP_WORDS,
    TFIDF_FILE,
    LDA_FILE,
    GLOVE_FILE,
    EPOCHS,
    BATCH_SIZE,
    PREDICTION_THRESHOLD,
)

from features_engineering import calculateTFIDFscore, calculateTopicProbability, loadSparseMatrix, saveSparseMatrix
from nb_svm import fitModel, NbSvmClassifier
from lstm import loadGloveEmbeddings, getWordVectors, getSequenceModel
from nb_svm import calculateModelMetrics


def main(args: argparse.Namespace) -> None:

    df = loadData(DATA_FILE)
    models = []
    files = []
    functions = []

    if args.train_bag_of_words:
        models.append("Bag of words")
        files.append(TFIDF_FILE)
        functions.append(calculateTFIDFscore)

    if args.train_topic_modeling:
        models.append("Topic modeling")
        files.append(LDA_FILE)
        functions.append(calculateTopicProbability)

    for model_name, file, function in zip(models, files, functions):

        if os.path.exists(file):
            features, _ = loadSparseMatrix(file)
        else:
            features, colnames = function(df[CONTENT])
            if args.save:
                saveSparseMatrix(file, features, colnames)

        X_train, X_test, y_train, y_test = train_test_split(features, df[LABELS], test_size=TEST_SIZE, random_state=123)
        model = NbSvmClassifier(C=C)
        _, metrics_test, measures = fitModel(model, X_train, X_test, y_train, y_test, LABELS)

        print(model_name + " performance on test set")
        print(pd.DataFrame(metrics_test, columns=LABELS, index=measures))
        print("")

    if args.train_lstm:

        word2vec = loadGloveEmbeddings(GLOVE_FILE)
        features, word2index = getWordVectors(df[CONTENT])
        model = getSequenceModel(word2index, word2vec, len(LABELS))

        X_train, X_test, y_train, y_test = train_test_split(features, df[LABELS], test_size=TEST_SIZE, random_state=123)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=int(BATCH_SIZE), shuffle=True)
        y_pred = model.predict(X_test) > PREDICTION_THRESHOLD

        metrics_test, measures = calculateModelMetrics(y_pred, y_test.values)

        print("LSTM performance on test set")
        print(pd.DataFrame(metrics_test, columns=LABELS, index=measures))
        print("")


def get_arg_parser():
    parser = argparse.ArgumentParser(description="This script addresses the problem of detecting toxic text content.")

    parser.add_argument("--save", action="store_true", help="Save model features")
    parser.add_argument("--train-bag-of-words", type=bool, default=False, help="Fitting bag of words algorithm")
    parser.add_argument("--train-topic-modeling", type=bool, default=False, help="Fitting topic modeling algorithm")
    parser.add_argument("--train-lstm", type=bool, default=True, help="Fitting bag of words algorithm")

    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)
