from nltk.stem import PorterStemmer


def porter_stemmer(tokens):
    """Word Stemming
    Returns the stems of each tokens

    Parameter:
    tokens = list of words
    list of tokenised words from the sentences

    Returns:
    --------
    Returns a list of stem corresponding to each words
    """

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
