#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
import torch
import random
import torch.backends.cudnn as cudnn

def is_true(x):
    return not not x and x.lower().startswith('t')

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--fold', type=str, default='train', help='Fold [default: train]')
parser.add_argument('--start_epoch', type=int, help='Epoch to start from (defaults to most recent epoch)')
parser.add_argument('--count', type=int, default=1, help='Number of counterfactuals to generate')
parser.add_argument('--image_size', type=int, default=32, help='image size')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
import gen_openmax
from networks import build_networks
from options import load_options

from folder import OpenSetImageFolder
from imagepreprocess import *


# TODO: Right now, to edit cf_speed et al, you need to edit params.json

start_epoch = options['start_epoch']
options = load_options(options)
options['epoch'] = start_epoch

if options['seed'] is not None:
    random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    cudnn.deterministic = True

# dataloader = CustomDataloader(**options)

traindir = os.path.join(options['data_dir'], 'train')
train_transforms, val_transforms, evaluate_transforms = preprocess_strategy(options['data_dir'].split('/')[-1],
                                                                            options['image_size'])

train_dataset = OpenSetImageFolder(
    traindir,
    train_transforms,
    seed=options['seed'],
    num_classes=options['num_classes'])

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=options['batch_size'], shuffle=True,
    num_workers=8, pin_memory=True, drop_last=True)

# Batch size must be large enough to make a square grid visual
options['batch_size'] = options['num_classes'] + 1

networks = build_networks(**options)

for i in range(options['count']):
    gen_openmax.generate(networks, dataloader, **options)
