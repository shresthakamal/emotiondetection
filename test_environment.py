import unittest

from emotiondetection.utils.stemmer import porter_stemmer
from emotiondetection.utils.lemmatiser import wordnet_lemmatizer
from emotiondetection.utils.text_processor import text_processor
from emotiondetection.utils.tfidfprocessor import TfidfProcessor
from emotiondetection.main import test_pipeline, train_pipeline


class TestProcessingPipeline(unittest.TestCase):
    def test_stemmer(self):
        self.assertEqual(
            porter_stemmer(["drinking", "drinked", "drink", "calculated", "learning"]),
            ["drink", "drink", "drink", "calcul", "learn"],
            "Test Failed !! => Stemming ",
        )

    def test_lemmatizer(self):
        self.assertEqual(
            wordnet_lemmatizer(
                ["drinking", "drinked", "drink", "calculated", "learning"]
            ),
            ["drink", "drink", "drink", "calculate", "learn"],
            "Test Failed !! => Lemmatization",
        )

    def test_text_processor(self):
        self.assertEqual(
            text_processor("It's good so far! Understood well…! <3 GOOD writing"),
            ["good", "far", "understood", "well", "good", "writing"],
            "Test Failed !! => Text Processor",
        )

    def test_tfidf_processor(self):
        processor = TfidfProcessor()
        corpus = [
            "days feel close partner friends feel peace also experience close contact people regard greatly",
            "every time imagine someone love could contact serious illness even death",
            "obviously unjustly treat possibility elucidate",
            "think short time live relate periods life think use short time",
            "realize direct feel discontent partner way try put blame instead sort feeliings",
        ]
        processor.create_vectorizer()
        test_matrix = processor.fit_transform_text(corpus)
        self.assertEqual(test_matrix.shape, (5, 37), "Test Failed !! =>TFIDF Processor")


class Test_ModelPipeline(unittest.TestCase):
    def test_test_pipeline(self):
        self.assertEqual(
            test_pipeline("multinominal_naive_bayes", "This is angry"),
            "anger",
            "Prediction Test Failed",
        )

        self.assertEqual(
            test_pipeline("multinominal_naive_bayes", "This is happy"),
            "joy",
            "Prediction Test Failed !!",
        )

    def test_train_pipeline(self):
        self.assertEqual(
            train_pipeline("multinominal_naive_bayes", split_ratio=0.2),
            True,
            "Model Training Failed !!",
        )


if __name__ == "__main__":

    unittest.main()


# Manchester City reached to the final of UEFA Champions League == FEAR
# All lives will be lost due to COVID-19 == SADNESS
# The reading contents in this course were very helpful for understanding the concepts == JOY
# The videos in the course were very fast paced were difficult to catch up with == FEAR