import os
import numpy as np
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import _pickle as cPickle
#_pickle was pickle earlier 

#nltk.download('all',download_dir='D:nltk_data')
# if u want to download all other dependencies.

nltk.download('averaged_perceptron_tagger',download_dir='D:/nlkt_data')
nltk.download('wordnet',download_dir='D:/nltk_data')
nltk.download('stopwords',download_dir='D:/nltk_data')
nltk.download('punkt',download_dir='D:/nltk_data')

