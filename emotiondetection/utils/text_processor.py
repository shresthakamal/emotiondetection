import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def text_processor(text):
    """Perform text processing
    This function takes input a raw text and performs basic preprocessing steps
    like converts text into lowecase, decntraction,
    removal of special characters, word tokenisation
    and removal of stop words

    Parameters:
    -----------
    text: String
    Raw sentences from the dataset

    Returns: List
    List of tokenised and processed words
    Processing includes:    Lowecase, Special Characters Removal, Tokenisation, Decontraction and Stopwords removals

    """
    # clean the words, remove symbols special chars
    text = text.lower()

    # Decontraction
    text = re.sub("n't", " not", text)
    text = re.sub("'re", " are", text)
    text = re.sub("'s", " is", text)
    text = re.sub("'ll", " will", text)
    text = re.sub("'ve", " have", text)
    text = re.sub("'m", " am", text)

    # Words Selection
    text = re.sub("[^A-Za-z]+", " ", text)

    # Word Tokenisation
    word_tokenized = word_tokenize(text)

    # Stop words removal
    stop_words = stopwords.words("english")

    stopwords_free = [word for word in word_tokenized if word not in stop_words]

    return stopwords_free
