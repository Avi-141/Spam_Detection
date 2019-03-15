import pandas
import csv
import os
import numpy as np
import sys
import string
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import _pickle as cPickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 

#nltk.path.append('D:/nltk_data/')
#nltk.download('averaged_perceptron_tagger',download_dir='D:/nlkt_data')
#nltk.download('all',download_dir='D:nltk_data')
#nltk.download('wordnet',download_dir='D:/nltk_data')
#nltk.download('stopwords',download_dir='D:/nltk_data')
#nltk.download('punkt',download_dir='D:/nltk_data')


# make_prediction(spam_detector, test_array)
# this can be added at the last

#cross_validation is depracated , but i wanted to test my code , so here it is.

table = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
stop_words = stopwords.words("english")

def load_dataset():
	try:
		data = pandas.read_csv("spam_dataset", sep="\t", names=["label", "message"])
		data["length"] = data["message"].map(lambda sms: len(sms))
		return data

	except Exception as e:
		print(e)
		sys.exit(1)

def tokenizer(message):
	message = message.lower() #should we use uniode(utf 8)?
	message = remove_punctuation(message)
	words = [ word for word in word_tokenize(message) if word not in stop_words ]
	WNL = WordNetLemmatizer()
	return [ WNL.lemmatize(word) for word in words ]

def remove_punctuation(string):
	return string.translate(table)

def build_naive_bayes_classifier(message_train, message_test, label_train, label_test):

	print(label_test)
	pipeline = Pipeline([
		("vector",CountVectorizer(analyzer=tokenizer)),
		("tfidf",TfidfTransformer()),
		("classifier", MultinomialNB())
	])

	scores = cross_val_score(pipeline, message_train, label_train, cv=10, scoring='accuracy', n_jobs=-1)
	print(scores)

	parameters = { 'tfidf__use_idf' : (True, False)}
	tuned_classifier = GridSearchCV(pipeline, parameters, n_jobs=-1, scoring='accuracy', cv= StratifiedKFold(label_train, n_folds=5))
	
	spam_detector = tuned_classifier.fit(message_train, label_train)

	return spam_detector

def make_prediction(estimator, array):
	print('Predictions are :\n')
	predictions = estimator.predict(array)
	for i in range(len(array)):
		print('{:<5}'.format(predictions[i]), ": ", array[i])
		print(estimator.predict_proba(array[i])[0])
	print

def main():
	data = load_dataset()
	message_train, message_test, label_train, label_test = train_test_split(data['message'],data['label'],test_size=0.1)

	try:
		try:
			spam_detector = cPickle.load(open("naive_bayes_spam_detector.pkl", "r"))
		except:
			spam_detector = build_naive_bayes_classifier(message_train, message_test, label_train, label_test)
		
		
		with open("naive_bayes_spam_detector.pkl", "wb") as detector:
			cPickle.dump(spam_detector, detector)

	except Exception as e:
		print(e)

if __name__ == '__main__':
	main()