# -*- coding: utf-8 -*-
import nltk
import regex as re

# Data loading
DATA_FILE = "data/train.csv"
TEST_SIZE = 0.2
PROCESSED_DATA_FILE = "data/train_clean.json"

# Data Labels
CONTENT = "comment_text"
UNIQUE_ID = "id"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Plotting
PRECISION = 1e4

# TfidfVectorizer
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
STRIP_ACCENTS = "unicode"
MAX_FEATURES = 10000
MIN_DF = 1
TFIDF_FILE = "data/word_counts.json"

# Logistic regression
LOG_REGRESSION_SOLVER = "lbfgs"  # Optimizer
MAX_ITER = 100  # Num of iterations
C = 4  # Inverse of regularization stregth

# LDA model
NUM_TOPICS = 15
LDA_FILE = "data/topic_probabaility.json"

# LSTM
DROPOUT_RATE = 0.5
EPOCHS = 50
BATCH_SIZE = 512
LSTM_HIDDEN_STATE = 128
DENSE_UNITS = 50
PREDICTION_THRESHOLD = 0.25
PATIENCE = 3                  # Number of epochs for early stopping

# Word Embeddings
GLOVE_FILE = "glove/glove.6B.50d.txt"
MAX_WORDS = 20000
MAX_SEQUENCE_LEN = 1000

# Text preprocessing
LEMMATIZER = nltk.stem.wordnet.WordNetLemmatizer()
PORTER = nltk.stem.PorterStemmer()
TOKENIZER = nltk.tokenize.RegexpTokenizer(r"\w+[']\w*|\w+")

# Tokens generated in text cleaning.
# Do not use any special characters
re.DEFAULT_VERSION = re.VERSION1  # Version of regex used for patterns
PATTERNS = [[] for i in range(6)]
PATTERNS[0] = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
PATTERNS[1] = re.compile("http://.*com")
PATTERNS[2] = re.compile("\d[19|20]\d{2}s?")
PATTERNS[3] = re.compile("\d+(?:st|nd|rd|th)")
PATTERNS[4] = re.compile("\d{1,3}([,]\d{3})*([.]\d+)*")
PATTERNS[5] = re.compile("[\d]+")

TOKENS = [[] for i in range(6)]
TOKENS[0] = "IP"
TOKENS[1] = "URL"
TOKENS[2] = "YEAR"
TOKENS[3] = "ORDER"
TOKENS[4] = "DIGIT"
TOKENS[5] = "DIGIT"

SPAM_TOKEN = "SPAM"
SPAM_CHAR_LIMIT = 50  # Longest english word: 45 chars


# Aphostrophes dict
# Credits: https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
APPO = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'd": "I had",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    "didn't": "did not",
    "tryin'": "trying",
}
