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
import wandb
import torch.nn as nn
import itertools
import time
from CVPR_code.CustomImageTextFolder import *
from CVPR_code.text_models import *
from torchmetrics.classification import ConfusionMatrix
import ssl
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report

_num_classes = 4

BASE_PATH = "/project/def-rmsouza/jocazar/"
TRAIN_DATASET_PATH = "train_set"
VAL_DATASET_PATH = "val_set"
TEST_DATASET_PATH = "test_set"


class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image


# Just to avoid errors in the loader
def get_dummy_pipeline():
    pipeline = A.Compose([
        A.Resize(width=320,
                 height=320,
                 interpolation=cv2.INTER_CUBIC),
        a_pytorch.transforms.ToTensorV2()
    ])

    return pipeline


def get_class_weights(train_dataset_path):

    train_set = CustomImageTextFolder(train_dataset_path)

    total_num_samples_dataset = 0.0
    num_samples_each_class = []
    for i in range(_num_classes):
        num_samples_each_class.append(len(train_set.per_class[i]))
        total_num_samples_dataset += (len(train_set.per_class[i]))

    class_weights = []

    for i in range(_num_classes):
        class_weight = total_num_samples_dataset / \
            (_num_classes * num_samples_each_class[i])
        class_weights.append(class_weight)

    return class_weights


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
        texts = data['text']

        input_token_ids = texts['tokens'].to(hw_device)
        attention_mask = texts['attention_mask'].to(hw_device)
        labels = labels.to(hw_device)

        model_outputs = model(_input_ids=input_token_ids,
                              _attention_mask=attention_mask)

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


def flatten(l):
    return [item for sublist in l for item in sublist]


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
            texts = data['text']

            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)
            labels = labels.to(device)

            # Inference
            outputs = model(_input_ids=input_token_ids,
                            _attention_mask=attention_mask)
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
        base_name = BASE_PATH+"model_weights/BEST_model_{}_FT_EPOCH_{}_LR_{}_Reg_{}_FractionLR_{}_OPT_{}_VAL_ACC_{:.3f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, args.fraction_lr, opt, val_acc)

    else:

        base_name = BASE_PATH+"model_weights/BEST_model_{}_epoch_{}_LR_{}_Reg_{}_VAL_ACC_{:.3f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, val_acc)

    base_name = base_name + "class_weights_{}".format(class_weights)
    base_name = base_name + ".pth"

    weights_path = base_name

    model.to("cpu")

    print("Saving weights to {}".format(weights_path))

    torch.save(model.state_dict(), weights_path)

    model.to(hw_device)


def count_parameters(model): return sum(p.numel() for p in model.parameters())


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]


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

    print("Text Model: {}".format(args.text_model))

    global_model = DistilBert(_num_classes, args.model_dropout)
    _batch_size = 32

    # 66 365 956
    if args.text_model == "distilbert":
        global_model = DistilBert(_num_classes, args.model_dropout)
        _batch_size = 64
    # 124 648 708
    elif args.text_model == "roberta":
        global_model = Roberta(_num_classes, args.model_dropout)
        _batch_size = 32
    # 109 485 316
    elif args.text_model == "bert":
        global_model = Bert(_num_classes, args.model_dropout)
        _batch_size = 32
    # 407 345 156 parameters
    elif args.text_model == "bart":
        global_model = Bart(_num_classes, args.model_dropout)
        _batch_size = 4
    else:
        print("Invalid Model: {}".format(args.text_model))
        sys.exit(1)

    print("Num total parameters of the model: {}".format(
        count_parameters(global_model)))
    print("Batch Size: {}".format(_batch_size))
    print("Learning Rate: {}".format(args.lr))
    print("Regularization Rate: {}".format(args.reg))
    print("Using class weights: {}".format(args.balance_weights))
    print("Optimizer: {}".format(args.opt))
    print("Grad Acc steps: {}".format(args.acc_steps))

    print("Training for {} epochs".format(args.epochs))
    if args.tl is True:
        print("Training for {} fine tuning epochs".format(args.ft_epochs))
        print("Fraction of the LR for fine tuning: {}".format(args.fraction_lr))

    config = dict(
        num_model_parameters=count_parameters(global_model),
        batch_size=_batch_size,
        learning_rate=args.lr,
        regularization=args.reg,
        balance_weights=args.balance_weights,
        optimizer=args.opt,
        batch_acc_steps=args.acc_steps,
        num_epochs=args.epochs,
        fine_tuning_epochs=args.ft_epochs,
        fraction_lr=args.fraction_lr,
        architecture=args.text_model,
        dataset_id="garbage",
    )

    run = wandb.init(
        project="Garbage Classification Text",
        config=config,
    )

    wandb.watch(global_model)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        global_model = nn.DataParallel(global_model)

    aux = [args.dataset_folder_name, TRAIN_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    train_dataset_path = root = os.path.join(BASE_PATH, dataset_folder)

    class_weights = get_class_weights(train_dataset_path)
    print("Class weights: {}".format(class_weights))

    _tokenizer = global_model.get_tokenizer()
    _max_len = global_model.get_max_token_size()

    aux = [args.dataset_folder_name, TRAIN_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    train_data = CustomImageTextFolder(
        root=os.path.join(BASE_PATH, dataset_folder),
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=get_dummy_pipeline()))

    aux = [args.dataset_folder_name, VAL_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    val_data = CustomImageTextFolder(
        root=os.path.join(BASE_PATH, dataset_folder),
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=get_dummy_pipeline()))

    # aux = [args.dataset_folder_name, VAL_DATASET_PATH]
    # dataset_folder = '_'.join(aux)
    # test_data = CustomImageTextFolder(
    #     root=os.path.join(BASE_PATH, dataset_folder),
    #     tokens_max_len=_max_len,
    #     tokenizer_text=_tokenizer,
    #     transform=Transforms(img_transf=get_dummy_pipeline()))

    _num_workers = 8

    data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=_batch_size,
                                                    shuffle=True,
                                                    num_workers=_num_workers,
                                                    pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                                  batch_size=_batch_size,
                                                  shuffle=True,
                                                  num_workers=_num_workers,
                                                  pin_memory=True)

    # data_loader_test = torch.utils.data.DataLoader(dataset=test_data,
    #                                                batch_size=_batch_size,
    #                                                shuffle=True,
    #                                                num_workers=_num_workers,
    #                                                pin_memory=True)

    print(f"Total num of texts: {len(train_data)}")
    for i in range(_num_classes):
        len_samples = len(train_data.per_class[i])
        print("Num of samples for class {}: {}. Percentage of dataset: {:.2f}".format(
            i, len_samples, (len_samples/len(train_data))*100))

    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(
            global_model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            global_model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        print("Invalid optimizer!")
        sys.exit(1)

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
                                                          len(data_loader_train.dataset),
                                                          device,
                                                          _batch_size,
                                                          optimizer,
                                                          class_weights,
                                                          args.balance_weights,
                                                          args.acc_steps)

        elapsed_time = time.time() - st
        print('Epoch time: {:.1f}'.format(elapsed_time))

        train_loss_avg = np.average(train_loss_per_batch)
        train_loss_history.append(train_loss_avg)

        print("Avg train loss on epoch {}: {:.3f}".format(epoch, train_loss_avg))
        print("Max train loss on epoch {}: {:.3f}".format(
            epoch, np.max(train_loss_per_batch)))
        print("Min train loss on epoch {}: {:.3f}".format(
            epoch, np.min(train_loss_per_batch)))

        global_model.eval()

        print("Starting train accuracy calculation for epoch {}".format(epoch))
        train_accuracy, _ = calculate_set_accuracy(global_model,
                                                   data_loader_train,
                                                   len(data_loader_train.dataset),
                                                   device,
                                                   _batch_size)

        print("Train set accuracy on epoch {}: {:.3f} ".format(
            epoch, train_accuracy))
        train_accuracy_history.append(train_accuracy)

        print("Starting val accuracy calculation for epoch {}".format(epoch))
        val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                          data_loader_val,
                                                          len(data_loader_val.dataset),
                                                          device,
                                                          _batch_size)

        print("Val set accuracy on epoch {}: {:.3f} ".format(
            epoch, val_accuracy))
        val_accuracy_history.append(val_accuracy)

        # print("Starting test accuracy calculation for epoch {}".format(epoch))
        # test_accuracy = calculate_set_accuracy(global_model,
        #                                       data_loader_test,
        #                                       len(data_loader_test.dataset),
        #                                       device,
        #                                       _batch_size)

        # print("Train set accuracy on epoch {}: {:.3f}".format(epoch, test_accuracy))
        # test_accuracy_history.append(test_accuracy)

        wandb.log({'epoch': epoch,
                   'epoch_time_seconds': elapsed_time,
                   'train_loss_avg': train_loss_avg,
                   'train_accuracy_history': train_accuracy,
                   'val_accuracy_history': val_accuracy,
                   'black_val_precision': val_report["black"]["precision"],
                   'blue_val_precision': val_report["blue"]["precision"],
                   'green_val_precision': val_report["green"]["precision"],
                   'ttr_val_precision': val_report["ttr"]["precision"]})

        if val_accuracy > max_val_accuracy:
            print("Best model obtained based on Val Acc. Saving it!")
            save_model_weights(global_model, args.text_model,
                               epoch, val_accuracy, device, False, args.balance_weights, args.opt)
            max_val_accuracy = val_accuracy
            best_epoch = epoch
        else:
            print("Not saving model on epoch {}, best Val Acc so far on epoch {}: {:.3f}".format(epoch, best_epoch,
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
                                                                    len(data_loader_train.dataset),
                                                                    device,
                                                                    _batch_size,
                                                                    optimizer,
                                                                    class_weights,
                                                                    args.balance_weights,
                                                                    args.acc_steps)
            elapsed_time = time.time() - st
            print('Fine Tuning: epoch time: {:.1f}'.format(elapsed_time))

            ft_train_loss_avg = np.average(ft_train_loss_per_batch)

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
            train_accuracy, _ = calculate_set_accuracy(global_model,
                                                       data_loader_train,
                                                       len(data_loader_train.dataset),
                                                       device,
                                                       _batch_size)

            print("Fine Tuning: train set accuracy on epoch {}: {:.3f} ".format(
                epoch, train_accuracy))
            train_accuracy_history.append(train_accuracy)

            print(
                "Fine Tuning: starting val accuracy calculation for epoch {}".format(epoch))
            val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                              data_loader_val,
                                                              len(data_loader_val.dataset),
                                                              device,
                                                              _batch_size)

            print("Fine Tuning: Val set accuracy on epoch {}: {:.3f}".format(
                epoch, val_accuracy))
            val_accuracy_history.append(val_accuracy)

            # print(
            #     "Fine Tuning: starting test accuracy calculation for epoch {}".format(epoch))
            # test_accuracy = calculate_set_accuracy(global_model,
            #                                       data_loader_test,
            #                                       len(data_loader_test.dataset),
            #                                       device,
            #                                       _batch_size)

            # print("Fine Tuning: Test set accuracy on epoch {}: {:.3f}".format(
            #     epoch, test_accuracy))
            # test_accuracy_history.append(test_accuracy)

            wandb.log({'epoch': epoch,
                       'epoch_time_seconds': elapsed_time,
                       'train_loss_avg': train_loss_avg,
                       'train_accuracy_history': train_accuracy,
                       'val_accuracy_history': val_accuracy,
                       'black_val_precision': val_report["black"]["precision"],
                       'blue_val_precision': val_report["blue"]["precision"],
                       'green_val_precision': val_report["green"]["precision"],
                       'ttr_val_precision': val_report["ttr"]["precision"]})

            if val_accuracy > max_val_accuracy:
                print("Fine Tuning: best model obtained based on Val Acc. Saving it!")
                save_model_weights(global_model, args.text_model,
                                   epoch, val_accuracy, device, True, args.balance_weights, args.opt)
                best_epoch = epoch
                max_val_accuracy = val_accuracy
            else:
                print("Fine Tuning: not saving model, best Val Acc so far on epoch {}: {:.3f}".format(best_epoch,
                                                                                                      max_val_accuracy))

    # Finished training, save data
    with open(BASE_PATH + 'save/train_loss_model_{}_LR_{}_REG_{}_class_weights_{}.csv'.format(
            args.text_model, args.lr, args.reg, args.balance_weights), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, train_loss_history))

    with open(BASE_PATH + 'save/train_acc_model_{}_LR_{}_REG_{}_class_weights_{}.csv'.format(
            args.text_model, args.lr, args.reg, args.balance_weights), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, train_accuracy_history))

    with open(BASE_PATH + 'save/val_acc_model_{}_LR_{}_REG_{}_class_weights_{}.csv'.format(
            args.text_model, args.lr, args.reg, args.balance_weights), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, val_accuracy_history))

    # Plot train loss
    train_loss_history = torch.FloatTensor(train_loss_history).cpu()
    plt.figure()
    plt.plot(range(len(train_loss_history)), train_loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.title('Model: {}'.format(args.text_model))
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_[REG]_{}_[OPT]_{}_class_weights_{}_train_loss.png'.format(
            args.text_model, args.epochs, args.lr, args.reg, args.opt, args.balance_weights))

    # Plot train accuracy
    train_accuracy_history = torch.FloatTensor(train_accuracy_history).cpu()
    plt.figure()
    plt.plot(range(len(train_accuracy_history)), train_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy')
    plt.title('Model: {}'.format(args.text_model))
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_[REG]_{}_[OPT]_{}_class_weights_{}_train_accuracy.png'.format(
            args.text_model, args.epochs, args.lr, args.reg, args.opt, args.balance_weights))

    # Plot val accuracy
    plt.figure()
    plt.plot(range(len(val_accuracy_history)), val_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Val accuracy per Epoch')
    plt.title('Model: {}'.format(args.text_model))
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_[REG]_{}_[OPT]_{}_class_weights_{}_val_accuracy.png'.format(
            args.text_model, args.epochs, args.lr, args.reg, args.opt, args.balance_weights))

    run.finish()
