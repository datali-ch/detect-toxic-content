import pandas as pd
import math
import nltk
import regex as re
import string
import json
import os.path
from pandas import DataFrame
from typing import List
from config import (
    APPO,
    TOKENS,
    PATTERNS,
    SPAM_TOKEN,
    SPAM_CHAR_LIMIT,
    CONTENT,
    STOP_WORDS,
    TEST_SIZE,
    PROCESSED_DATA_FILE,
    LEMMATIZER,
    PORTER,
    TOKENIZER,
)


def loadData(
    file: str, preprocess: bool = True, save_to_file: bool = True
) -> pd.DataFrame:
    """ Load train.csv dataset from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

        Args:
            file:                               data file stored locally, full or relative path
            preprocess (optional):              cleaning and tokenizing text data. Valid only when loading original file.
                                                True for preprocessing, false otherwise
            save_to_file (optional):            saving preprocessed data. Valid only when loading original file.
                                                True for saving, false otherwise

        Returns:
            data:                               df with comments and labels
    
    """

    if os.path.exists(PROCESSED_DATA_FILE):
        data = loadProcessedData(PROCESSED_DATA_FILE)
    else:
        data = pd.read_csv(file)
        if preprocess:
            data[CONTENT] = data[CONTENT].apply(lambda x: cleanAndTokenize(x))
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
            comment:                           text to be processed

        Returns:
            words:                             cleaned and tokenized comment
    """

    comment = comment.lower()

    for pattern, token in zip(PATTERNS, TOKENS):
        comment = pattern.sub(token, comment)

    tokens = set(TOKENS)
    words = TOKENIZER.tokenize(comment)

    idx = 0
    while idx < len(words):

        if words[idx] in APPO:
            words[idx] = APPO[words[idx]]
        elif words[idx] in STOP_WORDS:
            del words[idx]
            continue
        elif words[idx] not in tokens:
            if len(words[idx]) > SPAM_CHAR_LIMIT:
                words[idx] = SPAM_TOKEN
            else:
                words[idx] = re.sub(r"'", "", words[idx])
                words[idx] = LEMMATIZER.lemmatize(words[idx], "v")
                words[idx] = PORTER.stem(words[idx])

        idx += 1

    return " ".join(words)


def saveProcessedData(df: DataFrame, file: str) -> None:
    """ Save dataframe to json file

        Args:
            df:                                 dataframe to be saved
            file:                               file name to store df, full or relative path

        Returns:
            None
    """

    df.to_json(file)


def loadProcessedData(file: str) -> DataFrame:
    """ Load dataframe from json file

        Args:
            file:                               file name where data is stored, full or relative path
        
        Returns:
            df:                                 dataframe stored in file
    """

    return pd.read_json(file)
