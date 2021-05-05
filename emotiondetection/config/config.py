import os

# Training For AI Engineers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

DATASET_NAME = "ISEAR.csv"

DATASET_URL = "https://www.floydhub.com/api/v1/resources/qM4BHN3pNjkfkvYjMMtjU4/ISEAR.csv?content=true&rename=isearcsv"

MODEL_PATH = os.path.join(BASE_DIR, "models")

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")

EMOTIONS = {
    "anger": 1,
    "fear": 2,
    "joy": 3,
    "sadness": 4,
    "shame": 5,
    "disgust": 6,
    "guilt": 7,
}

REMOVED_EMOTIONS = ["disgust", "guilt"]

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

TEST_SIZE = 0.2
