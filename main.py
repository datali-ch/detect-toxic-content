import argparse
from load_data import loadData
import pandas as pd
from config import DATA_FILE, TEST_SIZE, CONTENT, LABELS, C
from sklearn.model_selection import train_test_split
from features_engineering import calculateTFIDFscore, calculateTopicProbability
from nb_svm import fitModel, NbSvmClassifier
from config import STOP_WORDS, C, DATA_FILE


def main(args: argparse.Namespace) -> None:
    """Main function
    """

	df = loadData(DATA_FILE)
	model_names = []
	features = []


	if fit_bag_of_words:
		word_count, features = calculateTFIDFscore(df[CONTENT])
		features.append(word_count)
		model_names.append("Bag of words")


	if fit_topic_modeling:
		topic_probabaility = calculateTopicProbability(df[CONTENT])
		features.append(topic_probabaility)
		model_names.append("Topic modeling")


	for model_name, feature in zip(models, features):	

		X_train, X_test, y_train, y_test = train_test_split(feature, df[LABELS], test_size = TEST_SIZE, random_state=123)
		model = NbSvmClassifier(C=C)
		_, metrics_test, measures = fitModel(model, X_train, X_test, y_train, y_test, LABELS)

		print(model_name + " performance on test set")
		print(pd.DataFrame(metrics_test, columns = LABELS, index = measures))
		print("")


def get_arg_parser():
    parser = argparse.ArgumentParser(description="This script addresses the problem of detecting toxic text content.")

    parser.add_argument('--save', action='store_true', help="Save model features")
    parser.add_argument('--train-bag-of-words', type=bool, default=True,
                        help="Fitting bag of words algorithm")
    parser.add_argument('--train-topic-modeling', type=bool, default=True,
                        help="Fitting topic modeling algorithm")

    return parser

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)