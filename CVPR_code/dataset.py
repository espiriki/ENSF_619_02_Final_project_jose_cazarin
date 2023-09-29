import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
from CustomImageTextFolder import CustomImageTextFolder
import albumentations as A
import numpy as np
from sklearn.model_selection import train_test_split as tts
import os

LABELS_COLUMN_NAME = "labels"
SAMPLES_COLUMN_NAME = "samples"


class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image


class TextDataset(Dataset):
    def __init__(self, labels, samples, tokenizer, max_len):
        self._samples = samples
        self._labels = labels
        self._tokenizer = tokenizer
        self._max_len = max_len

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        samples = str(self._samples[i])
        labels = self._labels[i]

        # https://huggingface.co/docs//main_classes/tokenizer
        encoding = self._tokenizer.encode_plus(
            samples,

            # If left unset or set to None, this will use the predefined
            # model maximum length if a maximum length is required by one
            # of the truncation/padding parameters. If the model has no
            # specific maximum input length (like XLNet) truncation/padding
            # to a maximum length will be deactivated.
            max_length=self._max_len,

            # False or 'do_not_truncate' (default):
            # No truncation (i.e., can output batch with sequence lengths
            # greater than the model maximum admissible input size).
            truncation=True,

            # Needed if there's padding
            # so the padding can be ignored
            return_attention_mask=True,


            return_token_type_ids=False,

            # Padding to self.max_len
            # Equivalent to the resizing of the images
            pad_to_max_length=True,

            # pt = pytorch
            return_tensors='pt'
        )

        # print("original text")
        # print(samples)
        # print("encoding")
        # print(encoding['input_ids'])
        # print("attention mask")
        # print(encoding['attention_mask'])
        # print("tokens")
        # print(self._tokenizer.tokenize(samples))
        # print("label")
        # print(labels)

        return {
            # labels
            LABELS_COLUMN_NAME: torch.tensor(labels, dtype=torch.long),

            # Os tokens estao aqui
            'tokens': encoding['input_ids'].flatten(),

            # attention mask para ignorar o padding
            'attention_mask': encoding['attention_mask'].flatten()
        }


def create_data_loader(df, _tokenizer, _max_len, _batch_size, _workers):

    # df.rename(columns={'msg': SAMPLES_COLUMN_NAME,
    #           'spam': LABELS_COLUMN_NAME}, inplace=True)

    df.rename(columns={'review': SAMPLES_COLUMN_NAME,
              'sentiment': LABELS_COLUMN_NAME}, inplace=True)

    # TextDataset equivale ao CustomImageFolder que eu criei
    # que usa o DatasetFolder
    dataset = TextDataset(
        samples=df[SAMPLES_COLUMN_NAME].to_numpy(),
        labels=df[LABELS_COLUMN_NAME].to_numpy(),
        tokenizer=_tokenizer,
        max_len=_max_len
    )

    return DataLoader(
        dataset,
        batch_size=_batch_size,
        num_workers=_workers
    )


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]


BASE_PATH = "D:\Mestrado\ENSF_619_02_Final_project_jose_cazarin"
TRAIN_DATASET_PATH = "train_set"
VAL_DATASET_PATH = "val_set"
TEST_DATASET_PATH = "test_set"

def image_and_text_loader(
        _workers,
        _batch_size,
        _max_len,
        _tokenizer,
        _image_pipeline):

    # folder_name = "try_set"
    folder_name = "winter_2023_dataset_with_text_original_resized"

    aux = [folder_name, TRAIN_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    train_data = CustomImageTextFolder(
        root=os.path.join(BASE_PATH, dataset_folder),
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=_image_pipeline))

    aux = [folder_name, VAL_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    val_data = CustomImageTextFolder(
        root=os.path.join(BASE_PATH, dataset_folder),
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=_image_pipeline))

    aux = [folder_name, TEST_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    test_data = CustomImageTextFolder(
        root=os.path.join(BASE_PATH, dataset_folder),
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=_image_pipeline))

    # sets = [Y_train_set, Y_validation_set, Y_test_set]
    # sets_names = ["Train", "Validation", "Test"]

    # for set, set_name in zip(sets, sets_names):
    #     num_samples_each_class = np.unique(set, return_counts=True)[1]
    #     total = np.sum(num_samples_each_class)
    #     print("{} set num of samples: {}".format(set_name, total))
    #     for i in range(4):
    #         print("    {} set percentage of class {}: {:.2f}".format(
    #             set_name,
    #             get_keys_from_value(all_data_img_text.class_to_idx, i),
    #             100*(num_samples_each_class[i]/total)))

    train_loader = DataLoader(
        train_data,
        batch_size=_batch_size,
        num_workers=_workers,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=_batch_size,
        num_workers=_workers,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=_batch_size,
        num_workers=_workers,
        shuffle=True
    )

    return train_loader, val_loader, test_loader
