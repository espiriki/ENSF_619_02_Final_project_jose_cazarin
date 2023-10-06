#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import csv
import os
import ssl
import time
from datetime import datetime

import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchmetrics.classification import ConfusionMatrix
from sklearn.metrics import classification_report

import albumentations as A
import albumentations.pytorch as a_pytorch

from models import *
from options import args_parser
from CustomImageTextFolder import *
import keep_aspect_ratio

_num_classes = 4

BASE_PATH = "/project/def-rmsouza/jocazar/"
TRAIN_DATASET_PATH = "Train"
VAL_DATASET_PATH = "Val"

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

class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image

def run_one_epoch(epoch_num, model, data_loader, len_train_data, hw_device,
                  batch_size, train_optimizer, weights, use_class_weights, acc_steps):

    batch_loss = []
    n_batches = math.ceil((len_train_data/batch_size))

    opt_weights = torch.FloatTensor(weights).cuda()

    if use_class_weights is True:
        criterion = torch.nn.CrossEntropyLoss(weight=opt_weights).to(hw_device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(hw_device)

    print("Using device: {}".format(hw_device))
    for batch_idx, (data, labels) in enumerate(data_loader):
        images = data['image']['raw_image']

        images, labels = images.to(hw_device), labels.to(hw_device)

        model_outputs = model(images)
        loss = criterion(model_outputs, labels)

        loss.backward()
        if acc_steps != 0:
            loss = loss / acc_steps

            if ((batch_idx + 1) % acc_steps == 0) or \
                    (batch_idx + 1 == len(data_loader)) or acc_steps == 0:
                # Update Optimizer
                print("Optimizer step on batch idx: {}".format(batch_idx))
                train_optimizer.step()
                train_optimizer.zero_grad()
        else:
            train_optimizer.step()
            train_optimizer.zero_grad()

        print("Batches {}/{} on epoch {}".format(batch_idx,
                                                 n_batches, epoch_num), end='\r')

        cpu_loss = loss.cpu()
        cpu_loss = cpu_loss.detach()
        batch_loss.append(cpu_loss)

    print("\n")

    return n_batches, batch_loss

def calculate_set_accuracy(
        model,
        data_loader,
        len_data,
        device,
        batch_size):

    n_batches = math.ceil((len_data/batch_size))

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        correct = 0
        for batch_idx, (data, labels) in enumerate(data_loader):
            images = data['image']['raw_image']
            images, labels = images.to(
                device), labels.to(device)

            # Inference
            outputs = model(images)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                          n_batches), end='\r')

            all_labels.append(labels.cpu())
            all_predictions.append(pred_labels.cpu())

        all_labels = flatten(all_labels)
        all_predictions = flatten(all_predictions)

        report = classification_report(all_labels, all_predictions,
                                       target_names=["black", "blue", "green", "ttr"], output_dict=True)

        print(report)

        acc = 100 * (correct/len_data)
        print("Set acc: ", acc)
        return acc, report

def save_model_weights(model, model_name, epoch_num, val_acc, hw_device, fine_tuning, class_weights, opt):

    if fine_tuning:
        base_name = "model_weights/BEST_model_{}_FT_EPOCH_{}_LR_{}_Reg_{}_Opt_{}_FractionLR_{}_VAL_ACC_{:.3f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, opt, args.fraction_lr, val_acc)

    else:

        base_name = "model_weights/BEST_model_{}_epoch_{}_LR_{}_Reg_{}_Opt_{}_VAL_ACC_{:.3f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, opt, val_acc)

    base_name = base_name + "class_weights_{}".format(class_weights)
    base_name = base_name + ".pth"

    weights_path = BASE_PATH + base_name

    model.to("cpu")

    print("Saving weights to {}".format(weights_path))

    torch.save(model.state_dict(), weights_path)

    model.to(hw_device)

def calculate_mean_std_train_dataset(train_dataset_path, pipeline):

    stats_train_data = CustomImageTextFolder(root=train_dataset_path,
                                             transform=Transforms(img_transf=pipeline))

    stats_loader = torch.utils.data.DataLoader(dataset=stats_train_data,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)

    channels_sum, std_sum, num_batches = 0, 0, 0

    for images, _ in stats_loader:
        images = images['image']['raw_image']

        channels_sum += torch.mean(images*1.0, dim=[0, 2, 3])
        std_sum += torch.std(images*1.0, dim=[0, 2, 3])/len(images)
        num_batches += 1

    mean = (channels_sum / num_batches)/255
    std = (std_sum / num_batches)/255

    return mean, std

def count_parameters(model): return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    args = args_parser()

    ssl._create_default_https_context = ssl._create_unverified_context

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    if args.tl is True:
        print("In Transfer Learning mode!!!")

    # This is to make results predictable, when splitting the dataset into train/val/test
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Image Model: {}".format(args.image_model))

    global_model = EffNetB4(_num_classes, args.tl).to(device)

    wandb.init(entity="kodexlab", project="snow_reflection", job_type="train",
               config=args)

    config = wandb.config
    config.update({"num_parameters": count_parameters(global_model)})

    # Logging image sizes
    config.update(
        {"image_input_size": eff_net_sizes[args.image_model]})

    # Loading dataset, calculating dataset statistics
    train_transform = keep_aspect_ratio.get_train_transform()
    mean, std = calculate_mean_std_train_dataset(
        BASE_PATH+TRAIN_DATASET_PATH, train_transform)
    print("mean: ", mean)
    print("std: ", std)

    # Class weights. 0=black, 1=blue, 2=green, 3=ttr
    class_weights = [2.1, 2.6, 3.0, 3.0]
    #class_weights = get_class_weights(BASE_PATH + TRAIN_DATASET_PATH, torch.FloatTensor([0, 1, 2, 3]))

    # Uncomment below to run over the model weights
    # weights = np.random.randn(4) * np.random.choice([1, 10], 4)
    # model.load_state_dict(torch.load('BEST_model_weights.pth'))
    if args.tl is False:
        model = global_model

        print("Model Info: ", model)

        if args.optimization == "SGD":

            train_optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.reg, momentum=args.momentum)
        elif args.optimization == "ADAM":

            train_optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.reg)

        elif args.optimization == "ADAGRAD":

            train_optimizer = torch.optim.Adagrad(
                model.parameters(), lr=args.lr, weight_decay=args.reg)

        # Learning rate scheduler
        fraction_lr = args.fraction_lr
        steps = math.ceil(fraction_lr*args.num_epochs)

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            train_optimizer, step_size=steps, gamma=0.1)

        # Sending everything to GPU
        print("Sending data to device ...")
        # Moved to class constructor
        #image_transforms = keep_aspect_ratio.get_train_transform(mean=mean, std=std)
        train_data = CustomImageTextFolder(root=BASE_PATH+TRAIN_DATASET_PATH,
                                           transform=Transforms(
                                               img_transf=keep_aspect_ratio.get_train_transform(mean=mean, std=std)),
                                           class_weights=class_weights)

        # Create validation set
        if args.tl is True:
            val_data = CustomImageTextFolder(root=BASE_PATH+VAL_DATASET_PATH,
                                             transform=Transforms(
                                                 img_transf=keep_aspect_ratio.get_val_transform(mean=mean, std=std)),
                                             class_weights=class_weights)
        else:
            # Create validation set
            val_data = CustomImageTextFolder(root=BASE_PATH+VAL_DATASET_PATH,
                                             transform=Transforms(
                                                 img_transf=keep_aspect_ratio.get_val_transform(mean=mean, std=std)))

        # Splitting the train dataset into train and val datasets
        len_train_data = len(train_data)
        val_size = int(0.2*len_train_data)
        train_size = len_train_data - val_size
        train_data, val_data = random_split(
            train_data, [train_size, val_size])

        print("len_train_data: ", len_train_data)

        # Setting up Dataloaders
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

        # Whether to use class weights during training
        if args.use_class_weights == "yes":
            use_class_weights = True
        else:
            use_class_weights = False

        if args.tl is False:
            print("Starting Training ...")
            wandb.watch(model)
            model = model.to(device)

            best_val_acc = 0.0
            all_train_loss = []
            all_val_loss = []
            wandb_logger = wandb.summary

            print("Starting training with number of epochs = {}".format(
                args.num_epochs))

            for epoch_num in range(args.num_epochs):

                print("Epoch {}/{}".format(epoch_num+1, args.num_epochs))

                model.train()
                n_batches, batch_loss = run_one_epoch(epoch_num,
                                                      model,
                                                      train_loader,
                                                      len_train_data,
                                                      device,
                                                      args.batch_size,
                                                      train_optimizer,
                                                      class_weights,
                                                      use_class_weights,
                                                      args.accumulation_steps)

                model.eval()

                val_acc, val_report = calculate_set_accuracy(
                    model, val_loader, val_size, device, args.batch_size)
                all_val_loss.append(np.mean(val_report['weighted avg']['f1-score']))
                wandb.log({"Validation F1 Score": np.mean(
                    val_report['weighted avg']['f1-score'])})

                wandb_logger({"Training Loss": batch_loss})
                wandb_logger({"Validation Loss": val_report})

                if val_acc > best_val_acc:

                    best_val_acc = val_acc
                    # Save the model weights
                    save_model_weights(model, args.image_model,
                                       epoch_num, best_val_acc, device, False, class_weights, args.optimization)

                if epoch_num > 1:

                    print("mean: ", mean)
                    print("std: ", std)

                    if epoch_num > 3:
                        last_val_accs = all_val_loss[-4:]
                        if abs(last_val_accs[0]-last_val_accs[-1]) < 1e-4:
                            break
                    # Learning rate scheduling
                    exp_lr_scheduler.step()
                    print("Updated lr: ", train_optimizer.param_groups[0]['lr'])
            # Save the best model weights
            print("Finished Training ...")

    elif args.tl is True:
        if args.image_model == "eff_v2_small":
            model = EffNetV2(model_name="efficientnetv2_rw_s",
                             num_classes=1000, in_channels=3)

            print("Transfer Learning with {}".format(args.image_model))
        elif args.image_model == "eff_v2_medium":
            model = EffNetV2(model_name="efficientnetv2_rw_m",
                             num_classes=1000, in_channels=3)
            print("Transfer Learning with {}".format(args.image_model))
        elif args.image_model == "eff_v2_large":
            model = EffNetV2(model_name="efficientnetv2_rw_l",
                             num_classes=1000, in_channels=3)
            print("Transfer Learning with {}".format(args.image_model))

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 4)

        if args.optimization == "SGD":

            train_optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.reg, momentum=args.momentum)
        elif args.optimization == "ADAM":

            train_optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.reg)

        elif args.optimization == "ADAGRAD":

            train_optimizer = torch.optim.Adagrad(
                model.parameters(), lr=args.lr, weight_decay=args.reg)

        val_data = CustomImageTextFolder(root=BASE_PATH+VAL_DATASET_PATH,
                                         transform=Transforms(
                                             img_transf=keep_aspect_ratio.get_val_transform(mean=mean, std=std)),
                                         class_weights=class_weights)

        val_loader = torch.utils.data.DataLoader(
            dataset=val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

        print("Loading weights from {}".format(
            BASE_PATH+"model_weights/BEST_model_{}_FT_EPOCH_{}_LR_{}_Reg_{}_Opt_{}_VAL_ACC_{:.3f}_class_weights_{}.pth".format(
                args.image_model, args.num_epochs, args.lr, args.reg, args.optimization, 85.467, class_weights)))

        model.load_state_dict(torch.load(BASE_PATH+"model_weights/BEST_model_{}_FT_EPOCH_{}_LR_{}_Reg_{}_Opt_{}_VAL_ACC_{:.3f}_class_weights_{}.pth".format(
            args.image_model, args.num_epochs, args.lr, args.reg, args.optimization, 85.467, class_weights)))

        print("Model loaded ...")

        val_acc, val_report = calculate_set_accuracy(
            model, val_loader, val_size, device, args.batch_size)

        # wandb.log({"Validation F1 Score": np.mean(val_report['weighted avg']['f1-score'])})
        print("Validation F1 Score: ", np.mean(
            val_report['weighted avg']['f1-score']))
