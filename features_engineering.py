# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pandas import DataFrame
from config import STOP_WORDS, STRIP_ACCENTS, MAX_FEATURES, MIN_DF, C, NUM_TOPICS, TFIDF_FILE, LDA_FILE
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from typing import List, Tuple
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
    features = np.array(tfv.get_feature_names())

    return word_counts, features


def saveTFIDFscore(word_count: csr_matrix, words: List[str], file: str) -> None:
    """ Save TF-IDF scores to json file

        Args:
            word_count (M,N):                   TF-IDF score for M examples and N words
            words (1,N):                        words for which TF-IDF score was calculated
            file:                               file name to store results, full or relative path

        Returns:
            None
    """

    json_content = dict()
    json_content["size"] = word_count.shape
    json_content["features"] = words
    word_count = word_count.astype(float)

    keys = word_count.todok().keys()
    keys = [tuple(map(int, key)) for key in keys]

    vals = word_count.todok().values()
    vals = list(map(float, vals))

    json_content["positions"] = keys
    json_content["counts"] = vals

    with open(file, "w") as f:
        json.dump(json_content, f, indent=4)


def loadTFIDFscore(file: str) -> Tuple[csr_matrix, List[str]]:
    """ Load TF-IDF scores from json file

        Args:
            file:                               file name where results are stored, full or relative path

        Returns:
            word_count (M,N):                   TF-IDF score for M examples and N words
            words (1,N):                        words for which TF-IDF score was calculated
    """
    
    with open(file, "rb") as f:
        json_content = json.load(f)

    word_count = sp.dok_matrix(json_content["size"], dtype=np.float64)

    for pos, count in zip(json_content["positions"], json_content["counts"]):
        word_count[pos[0], pos[1]] = count

    word_count = word_count.transpose().tocsr()

    return word_count, json_content["features"]


def calculateTopicProbability(df: DataFrame) -> csr_matrix:
    """ Calculates topic probability matrix using Latent Dirichlet Allocation

        Args:
            df:                               texts to score, one sample per row

        Returns:
            topic_probability:                topic (probability) distribution for texts in df
    """

    tokenized_text = df.apply(lambda x: split_words(x, remove_stopwords=False))
    dictionary = Dictionary(tokenized_text)

    corpus = [dictionary.doc2bow(text) for text in tokenized_text]
    ldamodel = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

    topic_probability = ldamodel[corpus]
    topic_probability = gensim.matutils.corpus2csc(topic_probability)

    return topic_probability.transpose()
