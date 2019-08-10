# -*- coding: utf-8 -*-

"""
This script addresses the problem of detecting toxic content.
Using data on negative online behaviour, it attempts to detect abuse and classify it
The full problem definition is described at:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/
Data needed can be downloaded under the same address.

IMPORTANT: Training machine learning models requires substantial computational resources.
           For LSTM model, make sure to adjust NUM_EPOCHS to your needs

Author: Magdalena Surowka
        Data Scientist | Machine Learning Specialist
        magdalena.surowka@gmail.com
"""

import argparse
from sklearn.model_selection import train_test_split
import os.path
from data_loading import loadData
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
    PATIENCE,
)
from keras.callbacks import EarlyStopping
from features_engineering import calculateTFIDFscore, calculateTopicProbability, loadSparseMatrix, saveSparseMatrix
from nb_svm import NbSvmClassifier
from lstm import loadGloveEmbeddings, getWordVectors, getSequenceModel
from nb_svm import calculateModelMetrics


def main(args: argparse.Namespace) -> None:

    df = loadData(DATA_FILE)

    if args.choose_model == 3:

        model_name = "LSTM"
        word2vec = loadGloveEmbeddings(GLOVE_FILE)
        features, word2index = getWordVectors(df[CONTENT])
        model = getSequenceModel(word2index, word2vec, len(LABELS))

    else:
        if args.choose_model == 1:
            file = TFIDF_FILE
            model_name = "Bag of words"
        elif args.choose_model == 2:
            file = LDA_FILE
            model_name = "Topic modeling"

        if os.path.exists(file):
            features, _ = loadSparseMatrix(file)
        else:
            features, colnames = calculateTFIDFscore(df[CONTENT])
            if args.save:
                saveSparseMatrix(file, features, colnames)
        model = NbSvmClassifier(C=C)

    X_train, X_test, y_train, y_test = train_test_split(
        features, df[LABELS].values, test_size=TEST_SIZE, random_state=123
    )

    if args.choose_model == 3:
        earlyStop = EarlyStopping(monitor="loss", verbose=0, patience=PATIENCE, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=1, batch_size=int(BATCH_SIZE), callbacks=[earlyStop], shuffle=True)
        y_pred = (model.predict(X_test, batch_size=int(BATCH_SIZE)) > PREDICTION_THRESHOLD).astype(int)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    metrics_test, measures = calculateModelMetrics(y_pred, y_test)

    print(model_name + " performance on test set")
    print(pd.DataFrame(metrics_test, columns=LABELS, index=measures))
    print("")


def get_arg_parser():
    parser = argparse.ArgumentParser(description="This script addresses the problem of detecting toxic text content.")

    parser.add_argument("--save", action="store_true", help="Save model features")
    parser.add_argument(
        "--choose-model",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Choose an algorithm: 1. Bag of Words, 2. Topic Modeling, 3. LSTM",
    )
    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)