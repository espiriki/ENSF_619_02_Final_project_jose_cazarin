#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-2,
                        help='regularization rate')
    parser.add_argument('--model_dropout', type=float, default=0.3,
                        help='model FC layer dropout')
    parser.add_argument(
        '--tl', action=argparse.BooleanOptionalAction, default=True, help="Whether to use transfer learning or not")
    parser.add_argument('--balance_weights',
                        action=argparse.BooleanOptionalAction, default=False, help="Whether to use class balance weights or not")
    parser.add_argument('--ft_epochs', type=int, default=5,
                        help='number of fine tuning epochs')
    parser.add_argument('--fraction_lr', type=float, default=5,
                        help='value to divide the regular LR for to use in fine tuning')
    parser.add_argument('--image_model', type=str, default='b4', help='model name')
    parser.add_argument('--text_model', type=str, default='distilbert', help='model name')

    parser.add_argument('--model_path', type=str, default="",
                        help='Model file to calculate accuracy against the test set. Must match the model architecture select with the -model parameter')

    parser.add_argument('--acc_steps', type=int, default=0,
                        help='Gradient accumulation steps')

    parser.add_argument('--opt', type=str, default="adamw",
                        help='Optimizer to use')

    parser.add_argument('--calculate_dataset_stats',
                        action=argparse.BooleanOptionalAction, default=False, help="Calculate the development set stats used for normalization")

    args = parser.parse_args()
    return args
