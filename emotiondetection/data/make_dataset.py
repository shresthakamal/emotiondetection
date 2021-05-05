from tqdm import tqdm
import os
import requests

from emotiondetection.config import config


def download_dataset(url):
    response = requests.get(conf.DATASET_URL, stream=True)

    file_path = os.path.join(conf.DATA_PATH, conf.DATASET_NAME)

    if os.path.exists(file_path):
        print("[ALERT] File already exists!!")
        return False

    with open(file_path, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)
    return True


if __name__ == "__main__":
    download_dataset(conf.DATASET_URL)
