import nltk
from emotiondetection.config import tfidf_parameters
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download("punkt")
# nltk.download("stopwords")


class TfidfProcessor:
    """Generates a TFIDF vectors for given text list
    This class creates the tfidf matrix for given list of texts

    Methods:

    class.create_vectorizer():
        Creates the vectorizer with the given input params
    Parameters:
    ------------
    Instance of Class

    class.transform_text():
    Parameters:
    -----------
    List
        List of text that needs to be vecotrised

    Returns:
    ---------
    2D matrix
        Rows = No of sentences in the given text list
        Columns = Max_features / Vocab Size defined in the processor

    """

    # Constructor
    def __init__(
        self,
        analyzer=tfidf_parameters.tfidf_params["analyzer"],
        vocab_size=tfidf_parameters.tfidf_params["vocab_size"],
        stop_words=tfidf_parameters.tfidf_params["stop_words"],
    ):
        self.analyzer = analyzer
        self.stop_words = stop_words
        self.vocab_size = vocab_size
        self.vectorizer = None

    # Regular Methods
    def create_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            max_features=self.vocab_size,
            stop_words=self.stop_words,
            analyzer=self.analyzer,
        )
        return True

    def fit_transform_text(self, text_list):
        text_matrix = self.vectorizer.fit_transform(text_list).toarray()
        return text_matrix

    def transform_text(self, text_list):
        text_matrix = self.vectorizer.transform(text_list)
        return text_matrix