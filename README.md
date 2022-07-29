#### Text Clustering*

Note:
* The code can be executed using Google Colab.

## Requirements

*The code was developed using Google Colab.*

Libraries:
* from collections import Counter, defaultdict
* !pip install contractions
* import contractions
* import gensim
* from gensim import corpora
* from gensim.corpora import Dictionary
* from gensim.models import LdaModel, Phrases
* from gensim.models.coherencemodel import CoherenceModel
* from gensim.models.doc2vec import TaggedDocument
* from gensim.utils import lemmatize
* from http.client import IncompleteRead
* !pip install lda-classification tomotopy
* from lda_classification.model import TomotopyLDAVectorizer
* import matplotlib.pyplot as plt
* import nltk
* from nltk import corpus
* from nltk import word_tokenize
* from nltk.corpus import stopwords
* from nltk.probability import FreqDist
* from nltk.stem.porter import PorterStemmer
* from nltk.stem.wordnet import WordNetLemmatizer
* from nltk.tokenize import RegexpTokenizer
* import numpy as np
* import pandas as pd
* from pprint import pprint
* !pip install pyLDAvis==2.1.2
* import pyLDAvis
* import pyLDAvis.gensim
* import random
* import re
* import scipy.cluster.hierarchy as sch
* import scipy.cluster.hierarchy as sch
* import seaborn as sns
* from sklearn import metrics
* from sklearn.metrics import homogeneity_score
* from sklearn.metrics import silhouette_score
* from sklearn import preprocessing
* from sklearn.cluster import AgglomerativeClustering, KMeans
* from sklearn.decomposition import LatentDirichletAllocation as LDA, PCA
* from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
* from sklearn.mixture import GaussianMixture
* from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split
* !pip install lda-classification tomotopy 
* from tomotopy import HDPModel
* from traitlets.traitlets import Dict
* import unicodedata
* from urllib import request
* from urllib.error import URLError
* !pip install wordcloud
* from wordcloud import WordCloud

## Usage

Given the required Python packages are installed, each section of the code can be executed.
