# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from scipy import sparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import upsetplot
from dython.nominal import theils_u
import random
import math
import nltk
import re
from config import (
    APPO,
    DIGIT_TOKEN,
    ORDER_TOKEN,
    SPAM_TOKEN,
    YEAR_TOKEN,
    SPAM_CHAR_LIMIT,
    IP_TOKEN,
    URL_TOKEN,
)


def getTopWordsByCategory(
    df, categories, word_counts, features, num_words=10, aggregate=False
):

    if aggregate:
        items = 1
    else:
        items = len(categories)

    counts = [0] * items
    words = [0] * items

    for i in range(items):

        if aggregate:
            rows2take = np.where(df[categories].sum(axis=1) > 0)[0]
        else:
            rows2take = df[categories[i]].nonzero()[0]

        curr_counts = word_counts[rows2take, :].toarray().mean(axis=0)

        idx2take = np.argsort(-curr_counts)[:num_words]
        counts[i] = curr_counts[idx2take]
        words[i] = features[idx2take]

    return words, counts


def plotTopWordsByCategory(words, counts, categories):

    COLORS = sns.color_palette()
    ROWS = math.ceil(len(categories) / 2)
    COLS = 2

    plt.figure(figsize=(16, 22))
    plt.suptitle("TF-IDF Top words per category", fontsize=20)
    gridspec.GridSpec(ROWS, COLS)

    for i in range(len(categories)):

        plt.subplot2grid((ROWS, COLS), (i // 2, i % 2))
        plotTopWords(words[i], counts[i], categories[i], color=COLORS[i], show=False)

    plt.show()


def plotTopWords(words, counts, label, color=None, show=True):

    if show:
        plt.figure(figsize=(18, 12))

    sns.barplot(words, counts, color=color)
    plt.title("Label: " + label, fontsize=15)
    plt.xlabel("Word", fontsize=12)
    plt.ylabel("TF-IDF score", fontsize=12)

    if show:
        plt.show()


def plotClassShares(labels, ratio):

    plt.figure(figsize=(15, 6))
    PRECISION = 1e4
    ax = sns.barplot(labels, ratio)

    plt.title("Share of different comment types", fontsize=20)
    plt.xlabel("Comment type", fontsize=14)
    plt.ylabel("Ratio of comment type", fontsize=14)

    for bar, val in zip(ax.patches, ratio):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            round(val * PRECISION) / PRECISION,
            ha="center",
            va="bottom",
        )

    plt.show()


def printSampleComments(df, content_label, class_labels, num):

    for label in class_labels:

        subset = df[content_label][df[label] == 1]
        num_cases = len(subset.index)
        iters = min(num_cases, num)

        print(label.upper() + ": ")
        for comment in range(iters):
            row2take = random.randint(0, num_cases)
            comment = subset.iloc[row2take]

            print(comment + "\n")

        print("")


def plotSetIntersections(df, labels, unique_id):

    df_subset = df[labels + [unique_id]]
    counts = df_subset.astype(bool).groupby(labels).count()[unique_id]
    upsetplot.plot(counts, subset_size="sum", show_counts="%d", sort_by="cardinality")
    plt.suptitle("Multiple tags per comment")
    plt.show()


def calculateUncertanityCoeff(df, labels):

    # Theil's U uncertanity coefficient
    # https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

    uncertanity_coeff = [[0] * len(labels) for label in labels]

    for label_1 in range(len(labels)):
        for label_2 in range(len(labels)):
            uncertanity_coeff[label_1][label_2] = theils_u(
                df[labels[label_1]], df[labels[label_2]]
            )

    return uncertanity_coeff


def plotUncertanityCoeff(coeff, labels):

    plt.figure(figsize=(10, 8))
    sns.heatmap(coeff, xticklabels=labels, yticklabels=labels, annot=True, square=True)
    plt.title("Theil's U coefficient", fontsize=20)
    plt.xlabel("X", fontsize=14)
    plt.xlabel("Y", fontsize=14)

    plt.show()
