from nltk.stem import WordNetLemmatizer


def wordnet_lemmatizer(tokens):
    """Returns the lemma version of any words
    Parameter:
    ---------
    List of words

    Returns:
    List of lemma corresponding to each words
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, pos="v") for token in tokens]
