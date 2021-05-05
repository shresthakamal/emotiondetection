import pickle
import os
from emotiondetection.config import config


def save_object(checkpoint_path, folder_name, file_name, object_arr):
    """"""

    print("[INFO: utils/save_object]: Saving file:{}".format(file_name))

    if not os.path.exists(os.path.join(config.BASE_DIR, "checkpoints", folder_name)):
        os.makedirs(os.path.join(config.BASE_DIR, "checkpoints", folder_name))

    outfile = open(os.path.join(checkpoint_path, folder_name, f"{file_name}.pkl"), "wb")

    pickle.dump(object_arr[0], outfile)

    return True


if __name__ == "__main__":
    obj = ["Test_Object"]
    print(save_object(config.CHECKPOINT_PATH, "folder_name", "file_name", obj[0]))
