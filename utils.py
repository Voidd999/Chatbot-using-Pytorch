import nltk  
import numpy as np  
from nltk.stem.porter import PorterStemmer 


def correct(a):
    # b = TextBlob(a)
    # correct_text = str(b.correct())
    return a.lower()
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def b_o_w(tokn_sentence, all_words):
    tokn_sentence = [stem(word) for word in tokn_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokn_sentence:
            bag[idx] = 1
    return bag
