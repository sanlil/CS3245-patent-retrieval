import nltk
import string
from nltk.stem.porter import *

__stemmer = PorterStemmer()
__use_stop_words = True
__stops = set(nltk.corpus.stopwords.words('english'))

# Some additional stop words specific to this corpus. The effect of stopping those words is still unclear.
__stops |= {'mechanism', 'technology', 'technique', 'using', 'means', 'apparatus', 'method', 'system', 'perform', 'include'}

def normalize(word):
    """
    Normalize a term by applying case-folding, stemming and stopping.
    If a term is stopped or otherwise needs to be ignored, None is returned.
    Otherwise the result of the normalization is returned.
    """
    if word in string.punctuation or (__use_stop_words and word in __stops):
        return None

    return __stemmer.stem(word.encode('utf-8').lower())
    