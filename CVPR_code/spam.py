
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_data_loader, image_and_text_loader
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer

from models import *
from transformers import BertModel
from torchvision import transforms
import albumentations.pytorch as a_pytorch
import albumentations as A
import warnings
import inspect
import math
import cv2

warnings.filterwarnings('ignore')

PRE_TRAINED_MODEL_NAME_DISTILBERT = 'distilbert-base-uncased'
PRE_TRAINED_MODEL_NAME_BERT = 'bert-base-uncased'
EPOCHS = 200
BATCH_SIZE = 24
NUM_WORKERS = 8
MAX_TOKEN_LEN = 128
PROB_AUG = 0.8
mean_train_dataset = [0.5558, 0.5318, 0.5029]
std_train_dataset = [0.0315, 0.0318, 0.0315]


def calculate_set_accuracy(
        model,
        data_loader,
        len_data,
        device,
        batch_size):

    n_batches = math.ceil((len_data/batch_size))

    with torch.no_grad():
        
        correct = 0
        for batch_idx, (data, labels) in enumerate(data_loader):

            images = data['image']['raw_image']
            texts = data['text']

            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)
            images, labels = images.to(
                device), labels.to(device)

            # Inference
            outputs = model(_input_ids=input_token_ids,
                            _attention_mask=attention_mask,
                            _images=images,
                            eval=True,
                            remove_text=True,
                            remove_image=False)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                          n_batches), end='\r')

        print("Only IMAGES test acc: ", 100 * (correct/len_data))

        correct = 0
        for batch_idx, (data, labels) in enumerate(data_loader):
            images = data['image']['raw_image']
            texts = data['text']

            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)
            images, labels = images.to(
                device), labels.to(device)

            # Inference
            outputs = model(_input_ids=input_token_ids,
                            _attention_mask=attention_mask,
                            _images=images,
                            eval=True,
                            remove_image=True,
                            remove_text=False)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                        n_batches), end='\r')

        print("Only TEXT test acc: ", 100 * (correct/len_data))

        correct = 0
        for batch_idx, (data, labels) in enumerate(data_loader):

            images = data['image']['raw_image']
            texts = data['text']

            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)
            images, labels = images.to(
                device), labels.to(device)

            # Inference
            outputs = model(_input_ids=input_token_ids,
                            _attention_mask=attention_mask,
                            _images=images,
                            eval=True,
                            remove_text=False,
                            remove_image=False)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                        n_batches), end='\r')

        print("BOTH test acc: ", 100 * (correct/len_data))

    return 0.0


def clean_txt(text):
    text = re.sub("'", "", text)
    text = re.sub("(\\W+)", " ", text)
    return text


def get_split(text1):
    l_total = []
    l_parcial = []
    if len(text1.split())//150 > 0:
        n = len(text1.split())//150
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:200]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_parcial))
    return l_total


def get_image_pipeline(_width, _height):

    normalize_transform = A.Normalize(mean=mean_train_dataset,
                                      std=std_train_dataset,
                                      always_apply=True)

    TRAIN_PIPELINE = A.Compose([
        A.Rotate(p=PROB_AUG, interpolation=cv2.INTER_CUBIC,
                 border_mode=cv2.BORDER_CONSTANT,
                 value=0, crop_border=True),
        A.Resize(width=_width,
                 height=_height,
                 interpolation=cv2.INTER_CUBIC),
        A.VerticalFlip(p=PROB_AUG),
        A.HorizontalFlip(p=PROB_AUG),
        A.RandomBrightnessContrast(p=PROB_AUG),
        A.Sharpen(p=PROB_AUG),
        A.Perspective(p=PROB_AUG,
                      pad_mode=cv2.BORDER_CONSTANT,
                      pad_val=0),
        # Using this transform just to zoom in an out
        A.ShiftScaleRotate(shift_limit=0, rotate_limit=0,
                           interpolation=cv2.INTER_CUBIC,
                           border_mode=cv2.BORDER_CONSTANT,
                           value=0, p=PROB_AUG,
                           scale_limit=0.3),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    return TRAIN_PIPELINE


def main():

    # This is to make results predictable, when splitting the dataset into train/val/test
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text_and_image_model = RobertaAndMBNet(4, 0.7)

    tokenizer = text_and_image_model.get_tokenizer()
    width, height = text_and_image_model.get_image_size()

    train_loader, val_loader, test_loader, total_data_len = image_and_text_loader(
        NUM_WORKERS,
        BATCH_SIZE,
        MAX_TOKEN_LEN,
        tokenizer,
        get_image_pipeline(width, height))

    train_optimizer = torch.optim.SGD(text_and_image_model.parameters(),
                                      lr=0.001/2)

    criterion = nn.CrossEntropyLoss().to(device)

    print(total_data_len)
    print("Starting training on device: {}".format(device))
    max_test_accuracy = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):

        text_and_image_model.train()
        text_and_image_model.to(device)

        batch_loss = []
        n_batches = len(train_loader)

        for batch_idx, (data, labels) in enumerate(train_loader):

            images = data['image']['raw_image']
            texts = data['text']

            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)
            images, labels = images.to(
                device), labels.to(device)

            # print(texts["original_text"])

            custom_outputs = text_and_image_model(_input_ids=input_token_ids,
                                                  _attention_mask=attention_mask,
                                                  _images=images)

            # Calculating Loss
            loss = criterion(custom_outputs, labels)
            loss.backward()
            train_optimizer.step()
            train_optimizer.zero_grad()

            print("Batches {}/{} on epoch {}".format(batch_idx,
                                                     n_batches, epoch), end='\r')
            cpu_loss = loss.cpu()
            cpu_loss = cpu_loss.detach()
            batch_loss.append(cpu_loss)

        # After epoch
        print("\n")
        train_loss_avg = np.average(batch_loss)
        print("Avg train loss on epoch {}: {:.3f}".format(epoch, train_loss_avg))
        print("Max train loss on epoch {}: {:.3f}".format(
            epoch, np.max(batch_loss)))
        print("Min train loss on epoch {}: {:.3f}".format(
            epoch, np.min(batch_loss)))

        text_and_image_model.eval()

        # print("Starting train accuracy calculation for epoch {}".format(epoch))
        # train_accuracy = calculate_set_accuracy(text_and_image_model,
        #                                         train_loader,
        #                                         len(train_loader.dataset),
        #                                         device,
        #                                         BATCH_SIZE)

        # print("Train set accuracy on epoch {}: {:.3f} ".format(
        #     epoch, train_accuracy))

        # print("Starting validation accuracy calculation for epoch {}".format(epoch))
        # val_accuracy = calculate_set_accuracy(text_and_image_model,
        #                                       val_loader,
        #                                       len(val_loader.dataset),
        #                                       device,
        #                                       BATCH_SIZE)

        # print("Val set accuracy on epoch {}: {:.3f}".format(epoch, val_accuracy))

        print("Starting test accuracy calculation for epoch {}".format(epoch))
        test_accuracy = calculate_set_accuracy(text_and_image_model,
                                               test_loader,
                                               len(test_loader.dataset),
                                               device,
                                               BATCH_SIZE)

        print("Test set accuracy on epoch {}: {:.3f}".format(epoch, test_accuracy))

        if test_accuracy > max_test_accuracy:
            print("Best model obtained based on Test Acc. Saving it!")
            max_test_accuracy = test_accuracy
            best_epoch = epoch
            text_and_image_model.to("cpu")

            weights_path = "./text_image_model_test_acc_{:.3f}_epoch_{}.model".format(
                max_test_accuracy, best_epoch)
            print("Saving weights to {}".format(weights_path))
            torch.save(text_and_image_model.state_dict(), weights_path)
            text_and_image_model.to(device)
        else:
            print("Not saving model on epoch {}, best Test Acc so far on epoch {}: {:.3f}".format(epoch, best_epoch,
                                                                                                  max_test_accuracy))


if __name__ == '__main__':
    main()
