#!/usr/bin/env python
import argparse
import os
import sys
import torch
import random
import torch.backends.cudnn as cudnn
from pprint import pprint

from folder import OpenSetImageFolder
from imagepreprocess import *

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
parser.add_argument('--aux_dataset', help='Path to aux_dataset file [default: None]')
parser.add_argument('--comparison_dataset', help='Optional comparison dataset for open set evaluation [default: None]')
parser.add_argument('--mode', default='', help='If set to "baseline" use the baseline classifier')
parser.add_argument('--image_size', type=int, default=32, help='image size')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from training import train_classifier
from networks import build_networks, save_networks, get_optimizers
from options import load_options, get_current_epoch
from comparison import evaluate_with_comparison
from evaluation import save_evaluation

options = load_options(options)

if options['seed'] is not None:
    random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    cudnn.deterministic = True

# dataloader = FlexibleCustomDataloader(fold='train', **options)
networks = build_networks(**options)
optimizers = get_optimizers(networks, finetune=True, **options)

# eval_dataloader = CustomDataloader(last_batch=True, shuffle=False, fold='test', **options)

# Data loading code
traindir = os.path.join(options['data_dir'], 'train')
valdir = os.path.join(options['data_dir'], 'val')
train_transforms, val_transforms, evaluate_transforms = preprocess_strategy(options['data_dir'].split('/')[-1],
                                                                            options['image_size'])

train_dataset = OpenSetImageFolder(
    traindir,
    train_transforms,
    seed=options['seed'],
    num_classes=options['num_classes'])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=options['batch_size'], shuffle=True,
    num_workers=8, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    OpenSetImageFolder(valdir, val_transforms, seed=options['seed'], num_classes=options['num_classes']),
    batch_size=options['batch_size'], shuffle=True,
    num_workers=8, pin_memory=True)

start_epoch = get_current_epoch(options['result_dir']) + 1
for epoch in range(start_epoch, start_epoch + options['epochs']):
    train_results = train_classifier(networks, optimizers, train_loader, epoch=epoch, **options)
    eval_results = evaluate_with_comparison(networks, val_loader, **options)
    print('[Epoch {}] errC {} errOpenSet {} ClosedSetAcc{}'
          .format(epoch,train_results['errC'],train_results['errOpenSet'],eval_results['classifier_closed_set_accuracy']))
    save_evaluation(eval_results, options['result_dir'], epoch)
    save_networks(networks, epoch, options['result_dir'])
