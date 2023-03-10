# Importing required libraries
import nltk  # Natural Language Toolkit
import numpy as np  # NumPy library for numerical operations
from nltk.stem.porter import PorterStemmer  # PorterStemmer for stemming
from textblob import TextBlob  # TextBlob library for text processing
import textblob  # Importing textblob for general text processing
from gingerit.gingerit import GingerIt  # Importing GingerIt for grammar correction

# Function to correct grammar using GingerIt
def correct(Sentence):
    corrected_text = GingerIt().parse(Sentence)
    return corrected_text['result']

# Function to check spelling using TextBlob
def checkSpelling(a):
    b = TextBlob(a)
    correct_text = str(b.correct())
    return correct_text

# Initializing PorterStemmer object for word stemming
stemmer = PorterStemmer()

# Function to tokenize sentence into individual words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Function to stem a given word using PorterStemmer
def stem(word):
    return stemmer.stem(word.lower())

# Function to convert tokenized sentence into bag of words
def b_o_w(tokn_sentence,all_words):
    tokn_sentence = [stem(word) for word in tokn_sentence]
    bag = np.zeros(len(all_words),dtype= np.float32)
    for idx,w in enumerate(all_words):
        if w in tokn_sentence:
            bag[idx] = 1
    return bag
