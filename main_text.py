import os
import sys
import math
import csv
import time
import ssl
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, SGD
from torchvision import transforms
import wandb
import keep_aspect_ratio
import albumentations as A
import albumentations.pytorch as a_pytorch
import cv2
import torchvision
from models import *
from options import args_parser
from CVPR_code.CustomImageTextFolder import CustomImageTextFolder
from CVPR_code.text_models import DistilBert, Roberta, Bert, Bart
from torchmetrics.classification import ConfusionMatrix

# Constants
_num_classes = 4
BASE_PATH = "/project/def-rmsouza/jocazar/"
TRAIN_DATASET_PATH = "train_set"
VAL_DATASET_PATH = "val_set"
TEST_DATASET_PATH = "test_set"

# Configuration settings
config = {
    'num_classes': _num_classes,
    'base_path': BASE_PATH,
    'train_dataset_path': TRAIN_DATASET_PATH,
    'val_dataset_path': VAL_DATASET_PATH,
    'test_dataset_path': TEST_DATASET_PATH,
    # Add other configuration settings here
}

# Define a class for transformations
class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image

# Function to calculate class weights
def get_class_weights(train_dataset_path):
    train_set = CustomImageTextFolder(train_dataset_path)
    total_num_samples_dataset = 0.0
    num_samples_each_class = []

    for i in range(_num_classes):
        num_samples_each_class.append(len(train_set.per_class[i]))
        total_num_samples_dataset += len(train_set.per_class[i])

    class_weights = [total_num_samples_dataset / (_num_classes * num_samples_each_class[i]) for i in range(_num_classes)]
    return class_weights

# Function to run one training epoch
def run_one_epoch(epoch_num, model, data_loader, len_train_data, hw_device,
                  batch_size, train_optimizer, weights, use_class_weights, acc_steps):
    batch_loss = []
    n_batches = math.ceil(len_train_data / batch_size)
    opt_weights = torch.FloatTensor(weights).cuda()

    if use_class_weights:
        criterion = CrossEntropyLoss(weight=opt_weights).to(hw_device)
    else:
        criterion = CrossEntropyLoss().to(hw_device)

    print(f"Using device: {hw_device}")

    for batch_idx, (data, labels) in enumerate(data_loader):
        texts = data['text']
        input_token_ids = texts['tokens'].to(hw_device)
        attention_mask = texts['attention_mask'].to(hw_device)
        labels = labels.to(hw_device)

        model_outputs = model(_input_ids=input_token_ids, _attention_mask=attention_mask)
        loss = criterion(model_outputs, labels)

        loss.backward()

        if acc_steps != 0:
            loss = loss / acc_steps

            if (batch_idx + 1) % acc_steps == 0 or (batch_idx + 1 == len(data_loader)) or acc_steps == 0:
                train_optimizer.step()
                train_optimizer.zero_grad()
        else:
            train_optimizer.step()
            train_optimizer.zero_grad()

        print(f"Batches {batch_idx}/{n_batches} on epoch {epoch_num}", end='\r')
        cpu_loss = loss.cpu().detach()
        batch_loss.append(cpu_loss)

    print("\n")
    return n_batches, batch_loss

# Function to calculate set accuracy
def calculate_set_accuracy(model, data_loader, len_data, device, batch_size):
    n_batches = math.ceil(len_data / batch_size)
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        correct = 0

        for batch_idx, (data, labels) in enumerate(data_loader):
            texts = data['text']
            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(_input_ids=input_token_ids, _attention_mask=attention_mask)
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print(f"Batches {batch_idx}/{n_batches}", end='\r')
            all_labels.append(labels.cpu())
            all_predictions.append(pred_labels.cpu())

        all_labels = [item for sublist in all_labels for item in sublist]
        all_predictions = [item for sublist in all_predictions for item in sublist]

        report = classification_report(all_labels, all_predictions, target_names=["black", "blue", "green", "ttr"], output_dict=True)
        print(report)

        acc = 100 * (correct / len_data)
        print(f"Set accuracy: {acc}")
        return acc, report

# Function to save model weights
def save_model_weights(model, model_name, epoch_num, val_acc, hw_device, fine_tuning, class_weights, opt):
    if fine_tuning:
        base_name = f"{BASE_PATH}model_weights/BEST_model_{model_name}_FT_EPOCH_{epoch_num+1}_LR_{args.lr}_Reg_{args.reg}_FractionLR_{args.fraction_lr}_OPT_{opt}_VAL_ACC_{val_acc:.3f}_"
    else:
        base_name = f"{BASE_PATH}model_weights/BEST_model_{model_name}_epoch_{epoch_num+1}_LR_{args.lr}_Reg_{args.reg}_VAL_ACC_{val_acc:.3f}_"

    base_name = f"{base_name}class_weights_{class_weights}.pth"
    weights_path = base_name
    model.to("cpu")

    print(f"Saving weights to {weights_path}")
    torch.save(model.state_dict(), weights_path)
    model.to(hw_device)

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Function to get keys from a dictionary value
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]

# Main function
if __name__ == '__main__':
    args = args_parser()
    ssl._create_default_https_context = ssl._create_unverified_context

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    if args.tl:
        print("In Transfer Learning mode!!!")

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Text Model: {args.text_model}")

    global_model = None
    _batch_size = None

    if args.text_model == "distilbert":
        global_model = DistilBert(_num_classes, args.model_dropout)
        _batch_size = 64
    elif args.text_model == "roberta":
        global_model = Roberta(_num_classes, args.model_dropout)
        _batch_size = 32
    elif args.text_model == "bert":
        global_model = Bert(_num_classes, args.model_dropout)
        _batch_size = 32
    elif args.text_model == "bart":
        global_model = Bart(_num_classes, args.model_dropout)
        _batch_size = 4
    else:
        print(f"Invalid Model: {args.text_model}")
        sys.exit(1)

    print(f"Num total parameters of the model: {count_parameters(global_model)}")
    print(f"Batch Size: {_batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Regularization Rate: {args.reg}")
    print(f"Using class weights: {args.balance_weights}")
    print(f"Optimizer: {args.opt}")
    print(f"Grad Acc steps: {args.acc_steps}")

    print(f"Training for {args.epochs} epochs")
    if args.tl:
        print(f"Training for {args.ft_epochs} fine tuning epochs")
        print(f"Fraction of the LR for fine tuning: {args.fraction_lr}")

    config['num_model_parameters'] = count_parameters(global_model)
    config['batch_size'] = _batch_size
    config['learning_rate'] = args.lr
    config['regularization'] = args.reg
    config['balance_weights'] = args.balance_weights
    config['optimizer'] = args.opt
    config['batch_acc_steps'] = args.acc_steps
    config['num_epochs'] = args.epochs
    config['fine_tuning_epochs'] = args.ft_epochs
    config['fraction_lr'] = args.fraction_lr
    config['architecture'] = args.text_model
    config['dataset_id'] = "garbage"

    run = wandb.init(
        project="Garbage Classification Text",
        config=config,
        name="Text model: " + str(args.text_model)
    )

    wandb.watch(global_model)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        global_model = torch.nn.DataParallel(global_model)

    aux = [args.dataset_folder_name, TRAIN_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    train_dataset_path = os.path.join(BASE_PATH, dataset_folder)

    class_weights = get_class_weights(train_dataset_path)
    print(f"Class weights: {class_weights}")

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

    _num_workers = 8

    data_loader_train = DataLoader(dataset=train_data,
                                   batch_size=_batch_size,
                                   shuffle=True,
                                   num_workers=_num_workers,
                                   pin_memory=True)

    data_loader_val = DataLoader(dataset=val_data,
                                 batch_size=_batch_size,
                                 shuffle=True,
                                 num_workers=_num_workers,
                                 pin_memory=True)

    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    if args.opt == "adamw":
        optimizer = AdamW(global_model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "sgd":
        optimizer = SGD(global_model.parameters(), lr=args.lr, weight_decay=args.reg)
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
        print(f'Epoch time: {elapsed_time:.1f}')

        train_loss_avg = np.average(train_loss_per_batch)
        train_loss_history.append(train_loss_avg)

        print(f"Avg train loss on epoch {epoch}: {train_loss_avg:.3f}")
        print(f"Max train loss on epoch {epoch}: {np.max(train_loss_per_batch):.3f}")
        print(f"Min train loss on epoch {epoch}: {np.min(train_loss_per_batch):.3f}")

        global_model.eval()

        print(f"Starting train accuracy calculation for epoch {epoch}")
        train_accuracy, _ = calculate_set_accuracy(global_model,
                                                   data_loader_train,
                                                   len(data_loader_train.dataset),
                                                   device,
                                                   _batch_size)

        print(f"Train set accuracy on epoch {epoch}: {train_accuracy:.3f}")
        train_accuracy_history.append(train_accuracy)

        print(f"Starting val accuracy calculation for epoch {epoch}")
        val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                          data_loader_val,
                                                          len(data_loader_val.dataset),
                                                          device,
                                                          _batch_size)

        print(f"Val set accuracy on epoch {epoch}: {val_accuracy:.3f}")
        val_accuracy_history.append(val_accuracy)

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
            print(f"Not saving model on epoch {epoch}, best Val Acc so far on epoch {best_epoch}: {max_val_accuracy:.3f}")

    print("Starting Fine tuning!!")

    if args.tl:
        for param in global_model.parameters():
            param.requires_grad = True

        for group in optimizer.param_groups:
            group['lr'] = args.lr/args.fraction_lr

        for epoch in range(args.ft_epochs):
            global_model.train()
            st = time.time()

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
            print(f'Fine Tuning: epoch time: {elapsed_time:.1f}')

            ft_train_loss_avg = np.average(ft_train_loss_per_batch)

            print(f"Fine Tuning: avg train loss on epoch {epoch}: {ft_train_loss_avg:.3f}")
            print(f"Fine Tuning: max train loss on epoch {epoch}: {np.max(ft_train_loss_per_batch):.3f}")
            print(f"Fine Tuning: min train loss on epoch {epoch}: {np.min(ft_train_loss_per_batch):.3f}")

            train_loss_history.append(ft_train_loss_avg)
            global_model.eval()

            print(f"Fine Tuning: starting train accuracy calculation for epoch {epoch}")
            train_accuracy, _ = calculate_set_accuracy(global_model,
                                                       data_loader_train,
                                                       len(data_loader_train.dataset),
                                                       device,
                                                       _batch_size)

            print(f"Fine Tuning: train set accuracy on epoch {epoch}: {train_accuracy:.3f}")
            train_accuracy_history.append(train_accuracy)

            print(f"Fine Tuning: starting val accuracy calculation for epoch {epoch}")
            val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                              data_loader_val,
                                                              len(data_loader_val.dataset),
                                                              device,
                                                              _batch_size)

            print(f"Fine Tuning: Val set accuracy on epoch {epoch}: {val_accuracy:.3f}")
            val_accuracy_history.append(val_accuracy)

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
                print(f"Fine Tuning: not saving model, best Val Acc so far on epoch {best_epoch}: {max_val_accuracy:.3f}")

    # Finished training, save data
    save_path = BASE_PATH + f'save/train_loss_model_{args.text_model}_LR_{args.lr}_REG_{args.reg}_class_weights_{args.balance_weights}.csv'
    with open(save_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(train_loss_history)

    save_path = BASE_PATH + f'save/train_acc_model_{args.text_model}_LR_{args.lr}_REG_{args.reg}_class_weights_{args.balance_weights}.csv'
    with open(save_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(train_accuracy_history)

    save_path = BASE_PATH + f'save/val_acc_model_{args.text_model}_LR_{args.lr}_REG_{args.reg}_class_weights_{args.balance_weights}.csv'
    with open(save_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(val_accuracy_history)

    # Plot train loss
    train_loss_history = torch.FloatTensor(train_loss_history).cpu()
    plt.figure()
    plt.plot(range(len(train_loss_history)), train_loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.title(f'Model: {args.text_model}')
    plot_path = BASE_PATH + f'save/[M]_{args.text_model}_[E]_{args.epochs}_[LR]_{args.lr}_[REG]_{args.reg}_[OPT]_{args.opt}_class_weights_{args.balance_weights}_train_loss.png'
    plt.savefig(plot_path)

    # Plot train accuracy
    train_accuracy_history = torch.FloatTensor(train_accuracy_history).cpu()
    plt.figure()
    plt.plot(range(len(train_accuracy_history)), train_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy')
    plt.title(f'Model: {args.text_model}')
    plot_path = BASE_PATH + f'save/[M]_{args.text_model}_[E]_{args.epochs}_[LR]_{args.lr}_[REG]_{args.reg}_[OPT]_{args.opt}_class_weights_{args.balance_weights}_train_accuracy.png'
    plt.savefig(plot_path)

    # Plot val accuracy
    plt.figure()
    plt.plot(range(len(val_accuracy_history)), val_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Val accuracy per Epoch')
    plt.title(f'Model: {args.text_model}')
    plot_path = BASE_PATH + f'save/[M]_{args.text_model}_[E]_{args.epochs}_[LR]_{args.lr}_[REG]_{args.reg}_[OPT]_{args.opt}_class_weights_{args.balance_weights}_val_accuracy.png'
    plt.savefig(plot_path)

    run.finish()
