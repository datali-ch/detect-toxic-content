import pandas as pd
from config import TEST_SIZE
import math
import nltk
import regex as re
import string
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
    CONTENT_LABEL,
    STOP_WORDS
)

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
porter = nltk.stem.PorterStemmer()


def loadData(file: str, preprocess: bool=True, sample_size: int = None, save_to_file: bool=True) -> pd.DataFrame:
    """ Load train.csv dataset from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

        Args:
            file(str):                          data file stored locally, full path
            sample_size(int, optional):         number of randomly sampled observations from file. If None,
                                                all observations will be included.
        Returns:
            train_set(pandas df):               df with N out of M observations from data file
            test_set(pandas df):                df with M-N observations from data file
    """

    data = pd.read_csv(file)
    if preprocess:
        data[CONTENT_LABEL] = data[CONTENT_LABEL].apply(lambda x: cleanAndTokenize(x))

    train_set, test_set = train_test_split(data, test_size=TEST_SIZE, random_state=666)
    
    return train_set, test_set

def cleanAndTokenize(comment: str) -> List[str]:

    # A: Clean:
    #   1. Lowercase
    #   2. Remove apostrophes
    #   3. Detect IP
    #   4. Detect links
    #   5. Detect years
    #   6. Detect digits
    #   7. Detect order (1st, 22nd etc.)
    #   8. Detect spam
    # B: Lemmatize
    # C: Stem


    re.DEFAULT_VERSION = re.VERSION1

    comment = comment.lower()

    pattern_ip = re.compile('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    pattern_url = re.compile('http://.*com')
    pattern_year = re.compile('\d[19|20]\d{2}s?')
    pattern_order = re.compile('\d+(?:st|nd|rd|th)') 
    pattern_digit = re.compile('[\d]+')
    pattern_clean = re.compile("[\W_--[ ']]+")

    comment = pattern_ip.sub(IP_TOKEN, comment)
    comment = pattern_url.sub(URL_TOKEN, comment)
    comment = pattern_year.sub(YEAR_TOKEN, comment)
    comment = pattern_order.sub(ORDER_TOKEN, comment)
    comment = pattern_digit.sub(DIGIT_TOKEN, comment)
    comment = pattern_clean.sub(' ', comment)
    
    words = comment.split()

    idx = 0
    while idx < len(words):

        if words[idx] in STOPWORDS:
            del words[idx]
            continue
        elif words[idx] in APPO:
            words[idx] = APPO[words[idx]]
        else:
            if len(words[idx]) > SPAM_CHAR_LIMIT:
                words[idx] = SPAM_TOKEN
            else:
                words[idx] = lemmatizer.lemmatize(words[idx], "v")
                words[idx] = porter.stem(words[idx])


        idx += 1

    return words