# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import upsetplot
from dython.nominal import theils_u
import random
import math
import re
from pandas import DataFrame
from typing import List, Tuple
from numpy import ndarray
from config import PRECISION


def getTopWordsByCategory(
    df: DataFrame,
    categories: List[str],
    word_counts: ndarray,
    features: List[str],
    n: int = 10,
    aggregate: bool = False,
) -> Tuple[List[str], List[int]]:
    """ Given word counts, returns top words in each class

        Args:
            df (N,M):                         M categories labels over N examples, one-hot encoding
            categories (1,M)                  names of M categories
            word_count (N,P):                 count of P words over N examples
            features (1,P):                   P words
            n:                                number of top words to return
            aggregate:                        indicates if words should be counted altogether or by category. True for aggregated results, false otherwise

        Returns:
            words (n, M) or (n, 1):           top n words
            counts (n, M) or (n, 1):          count of words in all df
    """

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

        idx2take = np.argsort(-curr_counts)[:n]
        counts[i] = curr_counts[idx2take]
        words[i] = [features[j] for j in idx2take]

    return words, counts


def plotTopWordsByCategory(words: List[str], counts: List[int], categories: List[str]) -> None:
    """ Plot histograms of top words in each category

        Args:
            words (N, M):                     top N words over M categories
            counts (N, M):                    count of words
            categories (1,M):                 names of M categories

        Returns:
            None
    """

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


def plotTopWords(words: List[str], counts: List[int], label: str, color: str = None, show: bool = True) -> None:
    """ Plot histogram of top words

        Args:
            words (n, 1):                     top n words
            counts (n, 1):                    count of words
            label:                            plot label
            color:                            plot color
            show:                             indicates if plot should be displayed. True for display, false otherwise

        Returns:
            None
    """

    if show:
        plt.figure(figsize=(18, 12))

    sns.barplot(words, counts, color=color)
    plt.title("Label: " + label, fontsize=15)
    plt.xlabel("Word", fontsize=12)
    plt.ylabel("TF-IDF score", fontsize=12)

    if show:
        plt.show()


def plotClassShares(labels: List[str], ratio: List[float]) -> None:
    """ Plot histogram of categories
        Args:
            labels:                         category names
            ratio:                          share of category over all examples

        Returns:
            None
    """

    plt.figure(figsize=(15, 6))
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


def printSampleComments(df: DataFrame, content_label: str, class_labels: List[str], n: int) -> None:
    """ Print sample comment
        Args:
            df:                         dataframe with text data
            content_label:              name of column with text data
            class_labels:               name of columns with labels, one-hot encoding
            n:                          number of comments to print

        Returns:
            None
    """

    for label in class_labels:

        subset = df[content_label][df[label] == 1]
        num_cases = len(subset.index)
        iters = min(num_cases, n)

        print(label.upper() + ": ")
        for comment in range(iters):
            row2take = random.randint(0, num_cases)
            comment = subset.iloc[row2take]

            print(comment + "\n")

        print("")


def plotSetIntersections(df: DataFrame, labels: List[str], unique_id: str) -> None:
    """ Plots sets size and intersection
        Args:
            df:                         dataframe with labels (one-hot encoding) and unique id
            class_labels:               name of columns with labels, one-hot encoding
            unique_id                   name of column with unique id

        Returns:
            None
    """

    df_subset = df[labels + [unique_id]]
    counts = df_subset.astype(bool).groupby(labels).count()[unique_id]
    upsetplot.plot(counts, subset_size="sum", show_counts="%d", sort_by="cardinality")
    plt.suptitle("Multiple tags per comment")
    plt.show()


def calculateUncertanityCoeff(df: DataFrame, labels: List[str]) -> List[List[float]]:
    """Calculates Theil's U uncertainity coefficient. Implemented as in:
        https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

        Args:
            df:                         dataframe one-hot encoding for M labels
            labels (1,M):               name of columns with M labels, one-hot encoding

        Returns:
            uncertanity_coeff (M,M):    theil's uncertanity coefficient for labels
    """

    uncertanity_coeff = [[0] * len(labels) for label in labels]

    for label_1 in range(len(labels)):
        for label_2 in range(len(labels)):
            uncertanity_coeff[label_1][label_2] = theils_u(df[labels[label_1]], df[labels[label_2]])

    return uncertanity_coeff


def plotUncertanityCoeff(coeff: List[List[float]], labels: List[str]) -> None:
    """ Plots uncertanity cofficient matrix

        Args:
            coeff (M,M):                Theil's U uncertanity coefficient for M classes
            labels (1,M):               names of M classes

        Returns:
            None
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(coeff, xticklabels=labels, yticklabels=labels, annot=True, square=True)
    plt.title("Theil's U coefficient", fontsize=20)
    plt.xlabel("X", fontsize=14)
    plt.xlabel("Y", fontsize=14)

    plt.show()
