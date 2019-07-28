from load_data import loadData
from config import DATA_FILE
import nltk
porter = nltk.stem.PorterStemmer()
lancaster=nltk.stem.LancasterStemmer()


df = loadData(DATA_FILE)
print("Loading successful")