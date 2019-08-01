# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pandas import DataFrame
from config import (
    STOP_WORDS,
    STRIP_ACCENTS,
    MAX_FEATURES,
    MIN_DF,
    C,
    NUM_TOPICS,
    TFIDF_FILE,
    LDA_FILE,
)
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from typing import List, Tuple, Union
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import json


def return_self(doc: List[str]) -> List[str]:
    """ Returns itself

        Args:
            doc:                            anything

        Returns
            doc:                            same anything
    """
    return doc


def split_words(doc: str, remove_stopwords: bool = False) -> List[str]:
    """ Tokenizes a string using whitespace

        Args:
            df:                               texts to score, one sample per row

        Returns:
            topic_probability:                topic probability distribution
    """

    doc = doc.split()
    if remove_stopwords:
        doc = [word for word in doc if not word in STOP_WORDS]
    return doc


def calculateTFIDFscore(
    df: DataFrame, ngrams: Tuple[int, int] = (1, 1)
) -> Tuple[csr_matrix, List[str]]:
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
        strip_accents=STRIP_ACCENTS,
        preprocessor=return_self,
        tokenizer=split_words,
        analyzer="word",
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
    )

    word_counts = tfv.fit_transform(df)
    features = tfv.get_feature_names()

    return word_counts, features


def calculateTopicProbability(df: DataFrame) -> Tuple[csr_matrix, None]:
    """ Calculates topic probability matrix using Latent Dirichlet Allocation

        Args:
            df:                               texts to score, one sample per row

        Returns:
            topic_probability:                topic (probability) distribution for texts in df
            features:                         dummy field for compatibility with calculateTFIDFscore 
    """

    tokenized_text = df.apply(lambda x: split_words(x, remove_stopwords=False))
    dictionary = Dictionary(tokenized_text)

    corpus = [dictionary.doc2bow(text) for text in tokenized_text]
    ldamodel = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

    topic_probability = ldamodel[corpus]
    topic_probability = gensim.matutils.corpus2csc(topic_probability)

    features = None

    return topic_probability.transpose(), features


def saveSparseMatrix(
    file: str, matrix: csr_matrix, colnames: Union[List[str], None] = None
) -> None:
    """ Save sparse matrix to json file

        Args:
            ile:                                file name to store results, full or relative path
            matrix (M,N):                       sparse matrix of size (M,N)
            colnames, optional (1,N):           column names

        Returns:
            None
    """

    json_content = dict()

    if colnames is not None:
        json_content["features"] = colnames

    json_content["size"] = matrix.shape
    matrix = matrix.astype(float)

    keys = matrix.todok().keys()
    keys = [tuple(map(int, key)) for key in keys]

    vals = matrix.todok().values()
    vals = list(map(float, vals))

    json_content["positions"] = keys
    json_content["counts"] = vals

    with open(file, "w") as f:
        json.dump(json_content, f, indent=4)


def loadSparseMatrix(file: str) -> Tuple[csr_matrix, Union[List[str], None]]:
    """ Load sparse matrix from json file

        Args:
            file:                               file name where matrix is stored, full or relative path

        Returns:
            matrix (M,N):                       sparse matrix of size (M,N)
            colnames (1,N):                     column names
    """

    with open(file, "rb") as f:
        json_content = json.load(f)

    matrix = sp.dok_matrix(tuple(json_content["size"]), dtype=np.float64)

    for pos, count in zip(json_content["positions"], json_content["counts"]):
        matrix[pos[0], pos[1]] = count

    if "features" in json_content:
        colnames = json_content["features"]
    else:
        colnames = None

    return matrix.tocsr(), colnames
