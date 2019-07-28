# -*- coding: utf-8 -*-
import nltk

# Labels
CONTENT_LABEL = "comment_text"
UNIQUE_ID = "id"
TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Data loading
DATA_FILE = 'data/train.csv'
TEST_SIZE = 0.2

# Parameters for TfidfVectorizer
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
STRIP_ACCENTS = 'unicode'
MAX_FEATURES = 10000
MIN_DF = 1

# Parameters for logistic regression
LOG_REGRESSION_SOLVER = "lbfgs" # Optimizer
C = 4                           # Penalty

# Tokens generated in text cleaning
DIGIT_TOKEN = "DIGIT"
ORDER_TOKEN = "ORDER"
SPAM_TOKEN = "SPAM"
YEAR_TOKEN = "YEAR"
SPAM_CHAR_LIMIT = (
    50
)  # Longest english word: 45 chars (pneumonoultramicroscopicsilicovolcanoconiosis)
IP_TOKEN = "IP"
URL_TOKEN = "URL"

# https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
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