import os
import pickle
import pandas as pd
import numpy as np
from emotiondetection.config import config
from emotiondetection.utils.text_processor import text_processor
from emotiondetection.utils.lemmatiser import wordnet_lemmatizer
from sklearn.model_selection import train_test_split
from emotiondetection.utils.tfidfprocessor import TfidfProcessor


def process_data(
    dataset_path, dataset_name, split_ratio=0.25, processor_engine="SKLEARN"
):
    """Process data with respect to the processor engine

    This function will process the text data from the given processor and return the
    vectorised forms of those data

    Parameters:
    -----------
    dataset_path: reponame/data/raw/
    dataset_name: ISEAR.csv
    split_ratio: Train-Test Split Ratio / Test_Size


    Returns:
    --------
    train_data
    y_train
    test_data
    y_test
    processor
    """

    df = pd.read_csv(
        os.path.join(dataset_path, dataset_name), names=["#", "emotions", "texts"]
    )

    df = df.loc[df["emotions"].apply(lambda x: x not in config.REMOVED_EMOTIONS)]

    df["emotions"] = df["emotions"].map(config.EMOTIONS)

    df = df.reset_index(drop=True)

    # Create a new column of the lemmatised columns
    df["lemmatised_tokens"] = generate_lemma(df, "texts")

    # Pickling Processed DataFrame
    processed_data_path = os.path.join(config.PROCESSED_DATA_PATH, "processed_data.pkl")
    if os.path.exists(processed_data_path):
        print("[INFO: build_features/process_data()]: DataFrame already pickled !!")
    else:
        df.to_pickle(processed_data_path)
        print(
            "[INFO: build_features/process_data()]: Data Pre-Processing Complete and Saved !!"
        )

    # Create Corpus and Target set
    corpus = df["lemmatised_tokens"].apply(" ".join).tolist()

    target = df["emotions"]

    x_train, x_test, y_train, y_test = train_test_split(
        corpus, target, test_size=split_ratio, random_state=53
    )

    if processor_engine == "SKLEARN":
        processor = TfidfProcessor()
        processor.create_vectorizer()
        x_train_vectors = processor.fit_transform_text(x_train)
        x_test_vectors = processor.transform_text(x_test)

        print(
            "[INFO: build_features/process_data]: Vectors: Train Size: {} & Test Size: {}".format(
                np.shape(x_train_vectors), np.shape(x_test_vectors)
            )
        )
    else:
        print("[ERROR]: Processor Engine Invalid !! ")

    return x_train_vectors, y_train, x_test_vectors, y_test, processor


def generate_lemma(df, col):
    """Generate lemma for given words
    This function takes the df and column containing processed sentences and
    generates lemma for each instance of the df[col]

    Parameters:
    ----------
    df: working DataFrame
    col: column containing processed texts
    """
    lemma = []
    for instance in df[col].tolist():
        tokens = text_processor(instance)
        lemma.append(wordnet_lemmatizer(tokens))
    return lemma


if __name__ == "__main__":

    x_train, y_train, x_text, y_text, processor = process_data(
        dataset_path=config.DATA_PATH,
        dataset_name=config.DATASET_NAME,
        split_ratio=config.TEST_SIZE,
        processor_engine="SKLEARN",
    )
