import pandas as pd
from config import TEST_SIZE

def loadData(file: str, sample_size: int=None): -> DataFrame:
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

    train_set, test_set = train_test_split(data, test_size=TEST_SIZE, random_state=666)

