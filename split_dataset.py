import numpy as np
from sklearn.model_selection import train_test_split as tts
from pathlib import Path
import os
import shutil
from CVPR_code.CustomImageTextFolder import *


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]


np.random.seed(42)
dataset_folder_name = "winter_2023_dataset_with_text_original_resized"

all_data_img_text = CustomImageTextFolder(root=dataset_folder_name)

X_train_set, X_val_plus_test_set, Y_train_set, Y_val_plus_test_set = tts(
    all_data_img_text,
    all_data_img_text.targets,
    # 80% for training
    test_size=0.2,
    stratify=all_data_img_text.targets
)

X_validation_set, X_test_set, Y_validation_set, Y_test_set = tts(
    X_val_plus_test_set,
    Y_val_plus_test_set,
    # From the rest, evenly divide between val and test set
    test_size=0.5,
    stratify=Y_val_plus_test_set,
)

sets = [Y_train_set, Y_validation_set, Y_test_set]
sets_names = ["Train", "Validation", "Test"]

print("Total samples: ", len(all_data_img_text))
for set, set_name in zip(sets, sets_names):
    num_samples_each_class = np.unique(set, return_counts=True)[1]
    total = np.sum(num_samples_each_class)
    print("{} set num of samples: {}".format(set_name, total))
    for i in range(4):
        print("    {} set num of samples of class {}: {}".format(
            set_name,
            get_keys_from_value(all_data_img_text.class_to_idx, i),
            (num_samples_each_class[i])))
        print("    {} set percentage of class {}: {:.2f}%".format(
            set_name,
            get_keys_from_value(all_data_img_text.class_to_idx, i),
            100*(num_samples_each_class[i]/total)))
        print("\n")

set_names = ["train_set", "val_set", "test_set"]
set_groups = [X_train_set, X_validation_set, X_test_set]
classes_idx = [0, 1, 2, 3]
for set_name, set_group in zip(set_names, set_groups):
    aux = [dataset_folder_name, set_name]
    for class_idx in classes_idx:
        class_ = get_keys_from_value(all_data_img_text.class_to_idx, class_idx)
        dest = os.path.join('_'.join(aux), class_)
        Path(dest).mkdir(parents=True, exist_ok=True)
    for image, label in set_group:
        file_path = image["image"]["image_path"]
        filename = os.path.basename(file_path)
        class_ = get_keys_from_value(all_data_img_text.class_to_idx, label)
        dest = os.path.join('_'.join(aux), class_)
        file_dest = os.path.join(dest, filename)
        shutil.copyfile(file_path, file_dest)
