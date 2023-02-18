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
import matplotlib.pyplot as plt
import math
import csv
import keep_aspect_ratio
import albumentations as A
import cv2
import albumentations.pytorch as a_pytorch
import numpy as np
# import wandb
import torch.nn as nn
import itertools
import time
from CustomImageFolder import CustomImageFolder
from torchmetrics.classification import ConfusionMatrix
import ssl

_num_classes = 4


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

TRAIN_DATA_PATH = BASE_PATH + "dataset_winter_2023_resized"


def run_one_epoch(epoch_num, model, data_loader, len_train_data, hw_device,
                  batch_size, train_optimizer, weights, use_class_weights):

    batch_loss = []
    n_batches = math.ceil((len_train_data/batch_size))

    opt_weights = torch.FloatTensor(weights).cuda()

    if use_class_weights is True:
        criterion = torch.nn.CrossEntropyLoss(weight=opt_weights).to(hw_device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(hw_device)

    print("Using device: {}".format(hw_device))
    for batch_idx, (images, labels) in enumerate(data_loader):

        images, labels = images.to(hw_device), labels.to(hw_device)

        train_optimizer.zero_grad()

        model_outputs = model(images)
        loss = criterion(model_outputs, labels)

        loss.backward()
        train_optimizer.step()

        print("Batches {}/{} on epoch {}".format(batch_idx,
                                                 n_batches, epoch_num), end='\r')

        cpu_loss = loss.cpu()
        cpu_loss = cpu_loss.detach()
        batch_loss.append(cpu_loss)

    print("\n")

    return n_batches, batch_loss


def calculate_train_accuracy(model, data_loader, len_train_data, hw_device, batch_size):

    correct = 0
    n_batches = math.ceil((len_train_data/batch_size))

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(data_loader):

            images, labels = images.to(hw_device), labels.to(hw_device)

            # Inference
            outputs = model(images)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                          n_batches), end='\r')

    print("\n")
    train_acc = 100 * (correct/len_train_data)
    return train_acc


def calculate_val_accuracy(model, data_loader, len_val_data, hw_device, batch_size):

    correct = 0
    n_batches = math.ceil((len_val_data/batch_size))

    all_preds = []
    all_labels = []
    confmat = ConfusionMatrix(task="multiclass", num_classes=_num_classes)
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

            print("Batches {}/{} ".format(batch_idx,
                                          n_batches), end='\r')

    all_preds = [item for sublist in all_preds for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]
    print(confmat(torch.tensor(all_labels), torch.tensor(all_preds)))

    val_acc = 100 * (correct/len_val_data)
    return val_acc


def save_model_weights(model, model_name, epoch_num, val_acc, hw_device, fine_tuning, class_weights):

    if fine_tuning:
        base_name = "model_weights/BEST_model_{}_FT_EPOCH_{}_LR_{}_Reg_{}_FractionLR_{}_VAL_ACC_{:.3f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, args.fraction_lr, val_acc)

    else:

        base_name = "model_weights/BEST_model_{}_epoch_{}_LR_{}_Reg_{}_VAL_ACC_{:.3f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, val_acc)

    base_name = base_name + "class_weights_{}".format(class_weights)
    base_name = base_name + ".pth"

    weights_path = BASE_PATH + base_name

    model.to("cpu")

    print("Saving weights to {}".format(weights_path))

    torch.save(model.state_dict(), weights_path)

    model.to(hw_device)


def calculate_mean_std_train_dataset(stats_train_data, pipeline):

    stats_train_data = CustomImageFolder(root=None, custom_samples=stats_train_data,
                                         transform=Transforms(img_transf=pipeline))

    stats_loader = torch.utils.data.DataLoader(dataset=stats_train_data,
                                               batch_size=32,
                                               shuffle=True, num_workers=8, pin_memory=True)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for _, (images, _) in enumerate(stats_loader):

        channels_sum += torch.mean(images*1.0, dim=[0, 2, 3])
        channels_squared_sum += torch.mean((images**2)*1.0, dim=[0, 2, 3])
        num_batches += 1

    mean = (channels_sum / num_batches)/255

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (torch.sqrt((channels_squared_sum / num_batches) - mean ** 2))/255

    return mean, std


if __name__ == '__main__':
    args = args_parser()

    ssl._create_default_https_context = ssl._create_unverified_context

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    if args.tl is True:
        print("In Transfer Learning mode!!!")

    config = dict(
        learning_rate=args.lr,
        architecture=args.model,
        regularization=args.reg,
        num_epochs=args.epochs,
        dataset_id="garbage",
    )

    # run = wandb.init(
    #     project="Garbage Classification",
    #     config=config,
    # )

    # This is to make results predictable, when splitting the dataset into train/val/test
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Model: {}".format(args.model))

    global_model = EffNetB4(_num_classes, args.tl)
    input_size = eff_net_sizes["b4"]
    _batch_size = 32
    if args.model == "b4":
        global_model = EffNetB4(_num_classes, args.tl)
        input_size = eff_net_sizes[args.model]
    elif args.model == "eff_v2_small":
        global_model = EffNetV2_S(_num_classes, args.tl)
        input_size = eff_net_sizes[args.model]
        _batch_size = 48
    elif args.model == "eff_v2_medium":
        global_model = EffNetV2_M(_num_classes, args.tl)
        input_size = eff_net_sizes[args.model]
    elif args.model == "eff_v2_large":
        global_model = EffNetV2_L(_num_classes, args.tl)
        input_size = eff_net_sizes[args.model]
        _batch_size = 8
    elif args.model == "b5":
        global_model = EffNetB5(_num_classes, args.tl)
        input_size = eff_net_sizes[args.model]
    elif args.model == "b7":
        global_model = EffNetB7(_num_classes, args.tl)
        input_size = eff_net_sizes[args.model]
        _batch_size = 6
    elif args.model == "b0":
        global_model = EffNetB0(_num_classes, args.tl)
        input_size = eff_net_sizes[args.model]
        _batch_size = 40
    elif args.model == "res18":
        global_model = ResNet18(_num_classes, args.tl)
        input_size = (300, 300)
    elif args.model == "res50":
        global_model = ResNet50(_num_classes, args.tl)
        input_size = (400, 400)
    elif args.model == "res152":
        global_model = ResNet152(_num_classes, args.tl)
        input_size = (500, 500)
    elif args.model == "next_tiny":
        global_model = ConvNextTiny(_num_classes, args.tl)
        input_size = (224, 224)
    elif args.model == "mb":
        global_model = MBNetLarge(_num_classes, args.tl)
        input_size = (320, 320)
    elif args.model == "vision":
        global_model = VisionLarge32(_num_classes, args.tl)
        input_size = (224, 224)
    elif args.model == "visionb":
        global_model = VisionB32(_num_classes, args.tl)
        input_size = (224, 224)
    else:
        print("Invalid Model: {}".format(args.model))
        sys.exit(1)

    print("Batch Size: {}".format(_batch_size))
    print("Learning Rate: {}".format(args.lr))
    print("Training for {} epochs".format(args.epochs))
    print("Use class weights: {}".format(args.balance_weights))
    if args.tl is True:
        print("Training for {} fine tuning epochs".format(args.ft_epochs))
        print("Fraction of the LR for fine tuning: {}".format(args.fraction_lr))

    print("Regularization Rate: {}".format(args.reg))

    print(global_model)

    # wandb.watch(global_model)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        global_model = nn.DataParallel(global_model)

    WIDTH = input_size[0]
    HEIGHT = input_size[1]
    AR_INPUT = WIDTH / HEIGHT

    STATS_PIPELINE = A.Compose([
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_CUBIC),
        a_pytorch.transforms.ToTensorV2()
    ])

    all_data_img_folder = CustomImageFolder(root=TRAIN_DATA_PATH)

    print("Total num of images: {}".format(len(all_data_img_folder)))
    for i in range(_num_classes):
        len_samples = len(all_data_img_folder.per_class[i])
        print("Num of samples for class {}: {}. Percentage of dataset: {:.2f}".format(
            i, len_samples, (len_samples/len(all_data_img_folder))*100))

    VALID_SPLIT = 0.90

    train_data_per_class = []
    val_data_per_class = []

    # Splitting the dataset while maintaining class proportions
    for all_data_current_class in all_data_img_folder.per_class:

        class_dataset_size = len(all_data_current_class)
        indices = torch.randperm(class_dataset_size).tolist()
        split_size = int(VALID_SPLIT*class_dataset_size)

        dataset_train = Subset(all_data_current_class, indices[:split_size])
        dataset_val = Subset(all_data_current_class, indices[split_size:])

        train_data_per_class.append(dataset_train)
        val_data_per_class.append(dataset_val)

    # Flatenning the list of lists (one list per class) into a single list
    train_data = list(itertools.chain.from_iterable(train_data_per_class))
    val_data = list(itertools.chain.from_iterable(val_data_per_class))
    class_weights = []

    # print("Common images: {}".format(len(list(set(train_data).intersection(val_data)))))

    print("Num of training images: {}".format(len(train_data)))
    for i in range(_num_classes):
        len_samples = len(train_data_per_class[i])
        print("Num of training samples for class {}: {}. Percentage of dataset: {:.2f}".format(
            i, len_samples, (len_samples/len(train_data))*100))
        this_weight = len(train_data)/(_num_classes * len_samples)
        class_weights.append(this_weight)

    print("Class weights: {}".format(class_weights))

    print("Num of validaton images: {}".format(len(val_data)))
    for i in range(_num_classes):
        len_samples = len(val_data_per_class[i])
        print("Num of validation samples for class {}: {}. Percentage of dataset: {:.2f}".format(
            i, len_samples, (len_samples/len(val_data))*100))

    # print("Calculating Train Dataset statistics...")
    # mean_train_dataset, std_train_dataset = calculate_mean_std_train_dataset(
    #     train_data, STATS_PIPELINE)

    mean_train_dataset = [0.5558, 0.5318, 0.5029]
    std_train_dataset = [0.0315, 0.0318, 0.0315]

    print("Mean Train Dataset: {}, STD Train Dataset: {}".format(
        mean_train_dataset, std_train_dataset))

    prob_augmentations = 0.6

    normalize_transform = A.Normalize(mean=mean_train_dataset,
                                      std=std_train_dataset, always_apply=True)

    TRAIN_PIPELINE = A.Compose([
        A.Rotate(p=prob_augmentations, interpolation=cv2.INTER_CUBIC,
                 border_mode=cv2.BORDER_CONSTANT,
                 value=0, crop_border=True),
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_CUBIC),
        A.VerticalFlip(p=prob_augmentations),
        A.HorizontalFlip(p=prob_augmentations),
        A.RandomBrightnessContrast(p=prob_augmentations),
        A.Sharpen(p=prob_augmentations),
        A.Perspective(p=prob_augmentations,
                      pad_mode=cv2.BORDER_CONSTANT,
                      pad_val=0),
        # Using this transform just to zoom in an out
        A.ShiftScaleRotate(shift_limit=0, rotate_limit=0,
                           interpolation=cv2.INTER_CUBIC,
                           border_mode=cv2.BORDER_CONSTANT,
                           value=0, p=prob_augmentations,
                           scale_limit=0.3),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    VALIDATION_PIPELINE = A.Compose([
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_CUBIC),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    train_data = CustomImageFolder(root=None, custom_samples=train_data,
                                   transform=Transforms(img_transf=TRAIN_PIPELINE))

    val_data = CustomImageFolder(root=None, custom_samples=val_data,
                                 transform=Transforms(img_transf=VALIDATION_PIPELINE))

    # cluster says the recommended ammount is 8
    _num_workers = 16

    data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=_batch_size,
                                                    shuffle=True, num_workers=_num_workers, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                                  batch_size=_batch_size,
                                                  shuffle=True, num_workers=_num_workers, pin_memory=True)

    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    optimizer = torch.optim.AdamW(
        global_model.parameters(), lr=args.lr, weight_decay=args.reg)

    print("Starting training...")
    global_model.to(device)
    max_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):

        global_model.train()
        st = time.time()

        num_batches, train_loss_per_batch = run_one_epoch(epoch,
                                                          global_model,
                                                          data_loader_train,
                                                          len(train_data),
                                                          device,
                                                          _batch_size,
                                                          optimizer,
                                                          class_weights,
                                                          args.balance_weights)

        elapsed_time = time.time() - st
        print('Epoch time: {:.1f}'.format(elapsed_time))

        train_loss_avg = np.average(train_loss_per_batch)
        train_loss_history.append(train_loss_avg)

        # wandb.log({'train_loss_avg': train_loss_avg,  'epoch': epoch})

        print("Avg train loss on epoch {}: {:.3f}".format(epoch, train_loss_avg))
        print("Max train loss on epoch {}: {:.3f}".format(
            epoch, np.max(train_loss_per_batch)))
        print("Min train loss on epoch {}: {:.3f}".format(
            epoch, np.min(train_loss_per_batch)))

        global_model.eval()

        print("Starting train accuracy calculation for epoch {}".format(epoch))
        train_accuracy = calculate_train_accuracy(global_model,
                                                  data_loader_train,
                                                  len(train_data),
                                                  device,
                                                  _batch_size)

        # wandb.log({'train_accuracy_history': train_accuracy,  'epoch': epoch})

        print("Train set accuracy on epoch {}: {:.3f} ".format(
            epoch, train_accuracy))
        train_accuracy_history.append(train_accuracy)

        print("Starting validation accuracy calculation for epoch {}".format(epoch))
        print(all_data_img_folder.class_to_idx)
        val_accuracy = calculate_val_accuracy(global_model,
                                              data_loader_val,
                                              len(val_data),
                                              device,
                                              _batch_size)

        # wandb.log({'val_accuracy_history': val_accuracy_history,  'epoch': epoch})

        print("Val set accuracy on epoch {}: {:.3f}".format(epoch, val_accuracy))
        val_accuracy_history.append(val_accuracy)

        if val_accuracy > max_val_accuracy:
            print("Best model obtained based on Val Acc. Saving it!")
            save_model_weights(global_model, args.model,
                               epoch, val_accuracy, device, False, args.balance_weights)
            max_val_accuracy = val_accuracy
            best_epoch = epoch
        else:
            print("Not saving model, best Val Acc so far on epoch {}: {:.3f}".format(best_epoch,
                                                                                     max_val_accuracy))

    print("Starting Fine tuning!!")
    # Fine tuning loop
    if args.tl is True:

        # set all model parameters to train
        for param in global_model.parameters():
            param.requires_grad = True

        # update learning rate of optimizer
        for group in optimizer.param_groups:
            group['lr'] = args.lr/args.fraction_lr

        for epoch in range(args.ft_epochs):

            global_model.train()
            st = time.time()
            # train using a small learning rate
            ft_num_batches, ft_train_loss_per_batch = run_one_epoch(epoch,
                                                                    global_model,
                                                                    data_loader_train,
                                                                    len(train_data),
                                                                    device,
                                                                    _batch_size,
                                                                    optimizer,
                                                                    class_weights,
                                                                    args.balance_weights)
            elapsed_time = time.time() - st
            print('Fine Tuning: epoch time: {:.1f}'.format(elapsed_time))

            ft_train_loss_avg = np.average(ft_train_loss_per_batch)

            # wandb.log({'train_loss_avg': train_loss_avg,  'epoch': epoch})

            print("Fine Tuning: avg train loss on epoch {}: {:.3f}".format(
                epoch, ft_train_loss_avg))
            print("Fine Tuning: max train loss on epoch {}: {:.3f}".format(
                epoch, np.max(ft_train_loss_per_batch)))
            print("Fine Tuning: min train loss on epoch {}: {:.3f}".format(
                epoch, np.min(ft_train_loss_per_batch)))

            train_loss_history.append(ft_train_loss_avg)
            global_model.eval()

            print(
                "Fine Tuning: starting train accuracy calculation for epoch {}".format(epoch))
            train_accuracy = calculate_train_accuracy(global_model,
                                                      data_loader_train,
                                                      len(train_data),
                                                      device,
                                                      _batch_size)

            # wandb.log({'train_accuracy_history': train_accuracy,  'epoch': epoch})

            print("Fine Tuning: train set accuracy on epoch {}: {:.3f} ".format(
                epoch, train_accuracy))
            train_accuracy_history.append(train_accuracy)

            print(
                "Fine Tuning: starting validation accuracy calculation for epoch {}".format(epoch))
            print(all_data_img_folder.class_to_idx)
            val_accuracy = calculate_val_accuracy(global_model,
                                                  data_loader_val,
                                                  len(val_data),
                                                  device,
                                                  _batch_size)
            print("Fine Tuning: Val set accuracy on epoch {}: {:.3f}".format(
                epoch, val_accuracy))

            val_accuracy_history.append(val_accuracy)

            if val_accuracy > max_val_accuracy:
                print("Fine Tuning: best model obtained based on Val Acc. Saving it!")
                save_model_weights(global_model, args.model,
                                   epoch, val_accuracy, device, True, args.balance_weights)
                best_epoch = epoch
                max_val_accuracy = val_accuracy
            else:
                print("Fine Tuning: not saving model, best Val Acc so far on epoch {}: {:.3f}".format(best_epoch,
                                                                                                      max_val_accuracy))

    # Finished training, save data
    with open(BASE_PATH + 'save/train_loss_model_{}_LR_{}_REG_{}_class_weights_{}.csv'.format(
            args.model, args.lr, args.reg, args.balance_weights), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, train_loss_history))

    with open(BASE_PATH + 'save/train_acc_model_{}_LR_{}_REG_{}_class_weights_{}.csv'.format(
            args.model, args.lr, args.reg, args.balance_weights), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, train_accuracy_history))

    with open(BASE_PATH + 'save/val_acc_model_{}_LR_{}_REG_{}_class_weights_{}.csv'.format(
            args.model, args.lr, args.reg, args.balance_weights), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, val_accuracy_history))

    # Plot train loss
    train_loss_history = torch.FloatTensor(train_loss_history).cpu()
    plt.figure()
    plt.plot(range(len(train_loss_history)), train_loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.title('Model: {}'.format(args.model))
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_[REG]_{}_class_weights_{}_train_loss.png'.format(
            args.model, args.epochs, args.lr, args.reg, args.balance_weights))

    # Plot train accuracy
    train_accuracy_history = torch.FloatTensor(train_accuracy_history).cpu()
    plt.figure()
    plt.plot(range(len(train_accuracy_history)), train_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy')
    plt.title('Model: {}'.format(args.model))
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_[REG]_{}_class_weights_{}_train_accuracy.png'.format(
            args.model, args.epochs, args.lr, args.reg, args.balance_weights))

    # Plot val accuracy
    plt.figure()
    plt.plot(range(len(val_accuracy_history)), val_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Val accuracy per Epoch')
    plt.title('Model: {}'.format(args.model))
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_[REG]_{}_class_weights_{}_val_accuracy.png'.format(
            args.model, args.epochs, args.lr, args.reg, args.balance_weights))

    # run.finish()
