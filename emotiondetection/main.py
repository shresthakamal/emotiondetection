from emotiondetection.config import config, model_params
from emotiondetection.models import train_model
from emotiondetection.features import build_features
from emotiondetection.models import train_model, test_model
from emotiondetection.utils.save_object import save_object
from sklearn.metrics import accuracy_score, f1_score
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
from emotiondetection.features.build_features import generate_lemma


def train_pipeline(model_name, split_ratio):
    """Train Pipeline function"""
    model_parameters = model_params.parameters[model_name]

    trainer = train_model.TrainModel.sklearn(model_name, **model_parameters)

    (
        train_data,
        train_label,
        test_data,
        test_label,
        processor,
    ) = build_features.process_data(
        dataset_path=config.DATA_PATH,
        dataset_name=config.DATASET_NAME,
        split_ratio=config.TEST_SIZE,
        processor_engine="SKLEARN",
    )

    save_status = save_object(
        config.CHECKPOINT_PATH, "processor", "processor", [processor]
    )

    kwargs = {"train_data": train_data, "train_label": train_label}

    clf = trainer.fit(**kwargs)

    prediction = clf.predict(test_data)

    accuracy_metric = accuracy_score(prediction, test_label)
    f1 = f1_score(prediction, test_label, average="weighted")

    print(
        "[INFO: main/train_pipeline] Accuracy: {}, F1 Score: {}".format(
            accuracy_metric, f1
        )
    )

    mlflow.log_metric("Accuracy", accuracy_metric)
    mlflow.log_metric("F1 Score", f1)
    mlflow.sklearn.log_model(clf, model_name)

    return True
    # Freexing is necessary here
    # Log the results using mlflow


def test_pipeline(model, input_text):
    """Test Pipeline Function"""

    df = pd.DataFrame([[input_text]], columns=["text"])

    df["input_lemma"] = generate_lemma(df, "text")

    corpus = df["input_lemma"].apply(" ".join).tolist()

    # Load Saved Model
    with open(
        os.path.join(config.CHECKPOINT_PATH, "SKLEARN", f"{model}.pkl"), "rb",
    ) as f:
        model = pickle.load(f)

    # Get Processor path to load the saved processor
    processor_path = os.path.join(config.CHECKPOINT_PATH, "processor", "processor.pkl")

    clf = test_model.TestModel.from_path(
        model=model, processor_path=processor_path, use_processor=True
    )

    prediction = clf.predict(corpus)
    # print(prediction)

    for key, value in config.EMOTIONS.items():
        if value == prediction[0]:
            emotion = key
            print("[RESULTS: (main/test_pipeline): Predicted Emotion: {}]".format(key))

    return emotion


if __name__ == "__main__":
    train_pipeline("multinominal_naive_bayes", split_ratio=0.2)
    test_pipeline("multinominal_naive_bayes", "It is too good to be bad.")
