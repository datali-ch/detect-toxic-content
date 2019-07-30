import pandas as pd
import math
import nltk
import regex as re
import string
import json
import os.path
from typing import List
from config import (
    APPO,
    DIGIT_TOKEN,
    ORDER_TOKEN,
    SPAM_TOKEN,
    YEAR_TOKEN,
    SPAM_CHAR_LIMIT,
    IP_TOKEN,
    URL_TOKEN,
    CONTENT,
    STOP_WORDS,
    TEST_SIZE,
    PROCESSED_DATA_FILE,
    LEMMATIZER, PORTER, TOKENIZER
)

def loadData(
    file: str, preprocess: bool = True, save_to_file: bool = True
) -> pd.DataFrame:
    """ Load train.csv dataset from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

        Args:
            file(str):                          data file stored locally, full path
            preprocess(bool, optional):         cleaning and tokenizing text data. Valid only when loading original file.
                                                True for preprocessing, false otherwise
            save_to_file(bool, optional):       saving preprocessed data. Valid only when loading original file.
                                                True for saving, false otherwise

        Returns:
            data(pandas df):                    df with comments and labels
    
    """
    if os.path.exists(PROCESSED_DATA_FILE):
        data = loadProcessedData(PROCESSED_DATA_FILE)
    else:
        data = pd.read_csv(file)
        if preprocess:
            data[CONTENT] = data[CONTENT].apply(
                lambda x: cleanAndTokenize(x)
            )
        if save_to_file:
            saveProcessedData(data, PROCESSED_DATA_FILE)

    return data


def cleanAndTokenize(comment: str) -> List[str]:
    """ Preprocess text for NLP applications. Preprocessing includes:
        1. Cleaning:
          - Lowercase
          - Remove apostrophes
          - Detect IP
          - Detect links
          - Detect years
          - Detect digits
          - Detect order (1st, 22nd etc.)
          - Detect spam
        2. Lemmatizing
        3. Stemming

        Args:
            comment(str):                       text to be processed

        Returns:
            words(List of str):                 cleaned and tokenized comment
    """

    re.DEFAULT_VERSION = re.VERSION1

    comment = comment.lower()

    PATTERNS = [[] for i in range(6)]
    PATTERNS[0] = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
    PATTERNS[1] = re.compile("http://.*com")
    PATTERNS[2] = re.compile("\d[19|20]\d{2}s?")
    PATTERNS[3]  = re.compile("\d+(?:st|nd|rd|th)")
    PATTERNS[4]  = re.compile("\d{1,3}([,]\d{3})*([.]\d+)*")
    PATTERNS[5]  = re.compile("[\d]+")

    tokens = [IP_TOKEN, URL_TOKEN, YEAR_TOKEN, ORDER_TOKEN, DIGIT_TOKEN, DIGIT_TOKEN]

    for pattern, token in zip(PATTERNS, tokens):
        comment = pattern.sub(token, comment)
    
    tokens = set(tokens)
    words = TOKENIZER.tokenize(comment)

    idx = 0
    while idx < len(words):

        if words[idx] in APPO:
            words[idx] = APPO[words[idx]]
        elif words[idx] not in tokens:
            if len(words[idx]) > SPAM_CHAR_LIMIT:
                words[idx] = SPAM_TOKEN
            else:
                words[idx] = re.sub(r"'", "", words[idx])
                words[idx] = LEMMATIZER.lemmatize(words[idx], "v")
                words[idx] = PORTER.stem(words[idx])

        idx += 1

    return ' '.join(words)


def saveProcessedData(df: pd.DataFrame, file: str) -> None:
    """ Save dataframe to json file

        Args:
            df(pandas df):                      dataframe to be saved
            file(str):                          file name to store df

        Returns:
            None
    """

    df.to_json(file)


def loadProcessedData(file: str) -> pd.DataFrame:
    """ Load dataframe from json file

        Args:
            file(str):                          file name where data is stored
        Returns:
            df(pandas df):                      dataframe stored in file
    """

    return pd.read_json(file)
