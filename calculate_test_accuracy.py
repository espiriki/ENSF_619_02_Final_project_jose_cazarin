#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from torchvision import transforms
import torchvision
from models import *
from options import args_parser
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
import torch
import math
import albumentations as A
import cv2
import albumentations.pytorch as a_pytorch
import numpy as np
# import wandb
import torch.nn as nn
import keep_aspect_ratio
from torchmetrics.classification import ConfusionMatrix


class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image


eff_net_sizes = {
    'b0': (256, 224),
    'b4': (384, 380),
    'b5': (489, 456),
    'b6': (561, 528),
    'b7': (633, 600),
    'eff_v2_small': (384, 384),
    'eff_v2_medium': (480, 480),
    'eff_v2_large': (480, 480)
}

BASE_PATH = "/project/def-rmsouza/jocazar/"

TEST_DATA_PATH = BASE_PATH + "test_dataset_fixed_resized_v2"


def calculate_test_accuracy(model, data_loader, len_test_data, hw_device, batch_size):

    correct = 0
    n_batches = math.ceil((len_test_data/batch_size))
    model.to(hw_device)
    all_preds = []
    all_labels = []
    confmat = ConfusionMatrix(task="multiclass", num_classes=4)
    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(data_loader):

            images, labels = images.to(hw_device), labels.to(hw_device)

            # Inference
            outputs = model(images)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            all_preds.append(pred_labels)
            all_labels.append(labels)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Test batches {}/{} ".format(batch_idx,
                                               n_batches))

            print("Running test accuracy: {:.3f} %".format(
                100*(correct/len_test_data)))

    print("\n")
    print("samples checked for test: {}".format(len_test_data))
    print("correct samples for test: {}".format(correct))
    test_acc = 100 * (correct/len_test_data)
    all_preds = [item for sublist in all_preds for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]

    print(confmat(torch.tensor(all_labels), torch.tensor(all_preds)))
    return test_acc


if __name__ == '__main__':
    args = args_parser()

    if args.model_path == "":
        print("Please provide test model path")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == "b4":
        global_model = EffNetB4()
        input_size = eff_net_sizes[args.model]
        _batch_size = 32
    elif args.model == "eff_v2_small":
        global_model = EffNetV2_S(4, args.tl)
        input_size = eff_net_sizes[args.model]
        _batch_size = 48
    elif args.model == "eff_v2_medium":
        global_model = EffNetV2_M(4, args.tl)
        input_size = eff_net_sizes[args.model]
    elif args.model == "eff_v2_large":
        global_model = EffNetV2_L(4, args.tl)
        input_size = eff_net_sizes[args.model]
    elif args.model == "b5":
        global_model = EffNetB5()
        input_size = eff_net_sizes[args.model]
        _batch_size = 16
    elif args.model == "b7":
        _batch_size = 8
        global_model = EffNetB7()
        input_size = eff_net_sizes[args.model]
    elif args.model == "b0":
        global_model = EffNetB0()
        input_size = eff_net_sizes[args.model]
        _batch_size = 40
    elif args.model == "res18":
        global_model = ResNet18()
        input_size = (300, 300)
    elif args.model == "res50":
        global_model = ResNet50()
        input_size = (400, 400)
    elif args.model == "res152":
        global_model = ResNet152()
        input_size = (500, 500)
    elif args.model == "next_tiny":
        global_model = ConvNextTiny()
        input_size = (224, 224)
    elif args.model == "mb":
        global_model = MBNetLarge()
        input_size = (320, 320)
    elif args.model == "vision":
        global_model = VisionLarge32()
        input_size = (224, 224)
    elif args.model == "visionb":
        global_model = VisionB32()
        input_size = (224, 224)
    else:
        print("Invalid Model: {}".format(args.model))
        sys.exit(1)

    print("Model: {}".format(args.model))

    # global_model = nn.DataParallel(global_model)

    model_name = args.model_path

    try:

        global_model.load_state_dict(torch.load(
            BASE_PATH + model_name))
    except:
        print("not correct model")

    global_model.eval()

    WIDTH = input_size[0]
    HEIGHT = input_size[1]
    AR_INPUT = WIDTH / HEIGHT

    mean_train_dataset = [0.5558, 0.5318, 0.5029]
    std_train_dataset = [0.0315, 0.0318, 0.0315]

    normalize_transform = A.Normalize(mean=mean_train_dataset,
                                      std=std_train_dataset, always_apply=True)

    TEST_PIPELINE = A.Compose([
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_CUBIC),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH,
                                                 transform=Transforms(img_transf=TEST_PIPELINE))

    print("Num of test images: {}".format(len(test_data)))

    # cluster says the recommended ammount is 8
    _num_workers = 8

    _batch_size = 32

    data_loader_test = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=_batch_size,
                                                   shuffle=True, num_workers=_num_workers, pin_memory=True)

    test_accuracy = calculate_test_accuracy(global_model,
                                            data_loader_test,
                                            len(test_data),
                                            device,
                                            _batch_size)

    print(test_data.class_to_idx)
    print("Test accuracy: {:.2f} %".format(test_accuracy))
