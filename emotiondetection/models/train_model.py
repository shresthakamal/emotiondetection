from emotiondetection.dispatcher import dispatcher
from emotiondetection.features import build_features
from emotiondetection.utils import save_object
from emotiondetection.config import config


class TrainModel:
    def __init__(self, model_name, engine, **kwargs):
        # kwargs contains the model specific arguments with train data and labels
        self.engine = engine
        self.model_name = model_name
        self.model = dispatcher.MODELS[model_name](**kwargs)

    def fit(self, **kwargs):
        # Havent specified which processor to use: SKELARN OR KERAS
        """Fits the model and save the model artifacts on the checkpoints

        Parameter:
        ----------
        Takes input the train_data and train_label

        Returns: clf
        Classifier

        Saves:
        Checkpoint of the trained model
        """
        train_data = kwargs["train_data"]
        train_label = kwargs["train_label"]

        print("[INFO:TrainModel.fit()]: Training on {} model".format(self.model_name))

        # We should also include the model parameters here
        clf = self.model.fit(train_data, train_label)

        status = save_object.save_object(
            config.CHECKPOINT_PATH, self.engine, self.model_name, [clf]
        )

        return clf

    @classmethod
    def sklearn(cls, model_name, **kwargs):
        return cls(model_name, engine="SKLEARN", **kwargs)

    @classmethod
    def keras(cls, model_name, **kwargs):
        return cls(model_name, engine="KERAS", **kwargs)


if __name__ == "__main__":
    trainer = TrainModel.sklearn(model_name="multinominal_naive_bayes")
