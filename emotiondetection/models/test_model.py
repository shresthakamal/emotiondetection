import os
import pickle
from emotiondetection.dispatcher import dispatcher


class TestModel:
    def __init__(self, model, processor, use_processor):
        self.model = model
        self.processor = processor
        self.use_processor = use_processor

    def predict(self, instances):
        if self.use_processor:
            test_data = instances
            processed_data = self.processor.transform_text(instances)
            prediction = self.model.predict(processed_data)

        return prediction

    @classmethod
    def from_path(cls, model, processor_path, use_processor=False):
        processor = None

        if use_processor:
            with open(processor_path, "rb") as handle:
                processor = pickle.load(handle)

        return cls(model, processor, use_processor)