from load_data import loadData
from config import DATA_FILE, TEST_SIZE, CONTENT, LABELS, C
from  sklearn.model_selection import train_test_split
import bag_of_words

df = loadData(DATA_FILE)

"""
# Bag of words
word_counts, _ = bag_of_words.calculateTFIDFscore(df[CONTENT])
X_train, X_test, y_train, y_test = train_test_split(word_counts, df[LABELS], test_size = TEST_SIZE, random_state=123)
model = bag_of_words.NbSvmClassifier(C=C)
metrics_train, metrics_test, measures = bag_of_words.fitModel(model, X_train, X_test, y_train, y_test, LABELS)

print(metrics_train)
"""

# Topic modeling
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

def dummy_toc(doc):
    return doc.split()

tokenized_text = df[CONTENT].apply(lambda x: dummy_toc(x))
# bigram = gensim.models.Phrases(tokenized_text)

# Remove stopwords
from config import STOP_WORDS
clean_text = [word for word in tokenized_text if word not in STOP_WORDS]
# clean_text = bigram[clean_text]

dictionary = Dictionary(clean_text)

corpus = [dictionary.doc2bow(text) for text in clean_text]
ldamodel = LdaModel(corpus=corpus, num_topics=15, id2word=dictionary)
topic_probability_mat = ldamodel[corpus]


