from load_data import loadData
from config import DATA_FILE, TEST_SIZE, CONTENT, LABELS, C
from  sklearn.model_selection import train_test_split
import bag_of_words

df = loadData(DATA_FILE)

word_counts, _ = bag_of_words.calculateTFIDFscore(df[CONTENT])
X_train, X_test, y_train, y_test = train_test_split(word_counts, df[LABELS], test_size = TEST_SIZE, random_state=123)
model = bag_of_words.NbSvmClassifier(C=C)
metrics_train, metrics_test, measures = bag_of_words.fitModel(model, X_train, X_test, y_train, y_test, LABELS)

print(metrics_train)



