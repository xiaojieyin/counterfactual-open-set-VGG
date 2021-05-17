#!/usr/bin/env python
import argparse
import os
import sys
import json
from pprint import pprint
import torch
import random
import torch.backends.cudnn as cudnn

from folder import OpenSetImageFolder
from imagepreprocess import *

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')
parser.add_argument('--epoch', default=None, type=int, help='Epoch to evaluate (latest epoch if none chosen)')
parser.add_argument('--comparison_dataset', type=str, help='Dataset for off-manifold comparison')
parser.add_argument('--aux_dataset', type=str, help='aux_dataset used in train_classifier')
parser.add_argument('--mode', default='', help='One of: default, weibull, weibull-kplus1, baseline')
parser.add_argument('--roc_output', type=str, help='Optional filename for ROC data output')
parser.add_argument('--image_size', type=int, default=32, help='image size')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import save_evaluation
from comparison import evaluate_with_comparison

options = load_options(options)
# if not options.get('epoch'):
#     options['epoch'] = get_current_epoch(options['result_dir'])
# TODO: Globally disable dataset augmentation during evaluation
options['random_horizontal_flip'] = False

# dataloader = CustomDataloader(last_batch=True, shuffle=False, **options)

# TODO: structure options in a way that doesn't require this sort of hack
# train_dataloader_options = options.copy()
# train_dataloader_options['fold'] = 'train'
# dataloader_train = CustomDataloader(last_batch=True, shuffle=False, **train_dataloader_options)


traindir = os.path.join(options['data_dir'], 'train')
valdir = os.path.join(options['data_dir'], 'val')
train_transforms, val_transforms, evaluate_transforms = preprocess_strategy(options['data_dir'].split('/')[-1],
                                                                            options['image_size'])

train_dataset = OpenSetImageFolder(
    traindir,
    train_transforms,
    seed=options['seed'],
    num_classes=options['num_classes'])

val_dataset = OpenSetImageFolder(
    valdir,
    val_transforms,
    seed=options['seed'],
    num_classes=options['num_classes'])

dataset_off = OpenSetImageFolder(
    valdir,
    val_transforms,
    seed=options['seed'],
    num_classes=options['num_classes'],
    fold='unknown')

dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=options['batch_size'], shuffle=False,
    num_workers=8, pin_memory=True, drop_last=False)

dataloader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=options['batch_size'], shuffle=False,
    num_workers=8, pin_memory=True, drop_last=False)

dataloader_off = torch.utils.data.DataLoader(
    dataset_off, batch_size=options['batch_size'], shuffle=False,
    num_workers=8, pin_memory=True, drop_last=False)

networks = build_networks(**options)

new_results = evaluate_with_comparison(networks, dataloader, comparison_dataloader=dataloader_off,
                                       dataloader_train=dataloader_train, **options)

pprint(new_results)
if not os.path.exists('evaluation'):
    os.mkdir('evaluation')
result = {options['mode']: new_results}
result_dir = os.path.join('evaluation', 'result_{}.json'.format(options['image_size']))
if os.path.exists(result_dir):
    old_result = json.load(open(result_dir))
    old_result.update(result)
    result = old_result.copy()
with open(result_dir, 'w') as fp:
    json.dump(result, fp, indent=2, sort_keys=True)

# save_evaluation(new_results, options['result_dir'], options['epoch'])
