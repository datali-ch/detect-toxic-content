import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Dropout, LSTM, Activation, CuDNNLSTM
from config import (
    DATA_FILE,
    CONTENT,
    LABELS,
    DROPOUT_RATE,
    LABELS,
    EPOCHS,
    BATCH_SIZE,
    LSTM_HIDDEN_STATE,
    GLOVE_FILE,
    TEST_SIZE,
    MAX_WORDS,
    MAX_SEQUENCE_LEN,
    PREDICTION_THRESHOLD,
)
from load_data import loadData
from keras.layers import Embedding, Input
from keras.models import Model
from typing import Dict, List, Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from nb_svm import calculateModelMetrics
import pandas as pd


def getSequenceModel(word2index: Dict, word2vec: Dict, num_categories: int) -> Model:
    """ Create graph model for multiclass classification using LSTM

        Args:
            word2index:                         dictionnary mapping words to indices
            word2vec:                           dictionnary mapping words to embedding vectors
            num_categories:                     number of classes

        Returns:
            model:                              graph model for LSTM with pretrained word embeddings
    """

    # Embedding layer
    sentence_indices = Input(shape=(MAX_SEQUENCE_LEN,), dtype="int32")
    embedding_layer = createEmbeddingLayer(word2index, word2vec)
    embeddings = embedding_layer(sentence_indices)

    # LSTM, Dropout and activation layers
    X = CuDNNLSTM(LSTM_HIDDEN_STATE, return_sequences=True)(embeddings)
    X = Dropout(rate=DROPOUT_RATE)(X)
    X = CuDNNLSTM(LSTM_HIDDEN_STATE, return_sequences=False)(X)
    X = Dropout(rate=DROPOUT_RATE)(X)
    X = Dense(num_categories, activation="sigmoid")(X)
    X = Activation("sigmoid")(X)
    model = Model(inputs=sentence_indices, outputs=X)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def getWordVectors(df: DataFrame) -> Tuple[List[int], Dict]:
    """ Convert the sentence into word vector representation

        Args:
            df:                                 text data, one sample per row

        Returns:
            data:                               text data represented with indices
            word2index:                         dictionnary mapping words to indices
    """

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df)
    word2index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(df)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN)

    return data, word2index


def loadGloveEmbeddings(file: str) -> Dict:
    """ Load GloVe word embeddings from a file

        Args:
            file:                               file with GloVe embeddings. File format as in https://nlp.stanford.edu/projects/glove/

        Returns:
            word2vec:                           dictionnary mapping word to empedding vector
    """

    word2vec = {}
    f = open(GLOVE_FILE, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        word2vec[word] = coefs

    return word2vec


def createEmbeddingLayer(word2index: dict, word2vec: dict) -> Embedding:
    """ Set up pretrained Embedding layer
        Credits to: https://keras.io/examples/pretrained_word_embeddings/

        Args:
            word2index:                         dictionnary mapping words to indices
            word2vec:                           dictionnary mapping words to embedding vectors

        Returns:
            embedding_layer                     layer with pretrained GloVe word embeddings
    """

    num_words = min(MAX_WORDS, len(word2index)) + 1
    emb_dim = word2vec["cucumber"].shape[0]
    emb_matrix = np.zeros((num_words, emb_dim))

    for word, index in word2index.items():
        if index > MAX_WORDS:
            break
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector

    embedding_layer = Embedding(
        num_words, emb_dim, input_length=MAX_SEQUENCE_LEN, trainable=False
    )  # Do not update word embeddings
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer