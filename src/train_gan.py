#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint
import time
import tabulate
import torch
import random
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
parser.add_argument('--aux_dataset', help='Path to aux_dataset file [default: None]')
parser.add_argument('--mode', default='', help='One of: default, weibull, weibull-kplus1, baseline')
parser.add_argument('--image_size', type=int, default=32, help='image size')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from training import train_gan
from networks import build_networks, save_networks, get_optimizers, save_best_networks
from options import load_options, get_current_epoch
from counterfactual import generate_counterfactual
from comparison import evaluate_with_comparison

from folder import OpenSetImageFolder
from imagepreprocess import *

options = load_options(options)

if options['seed'] is not None:
    random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    cudnn.deterministic = True

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

networks = build_networks(**options)
optimizers = get_optimizers(networks, **options)

start_epoch = get_current_epoch(options['result_dir']) + 1
best_acc = 0.0
best_epoch = 0
for epoch in range(start_epoch, start_epoch + options['epochs']):
    time_ep = time.time()

    train_results = train_gan(networks, optimizers, train_loader, epoch=epoch, **options)
    # generate_counterfactual(networks, dataloader, **options)
    eval_results = evaluate_with_comparison(networks, val_loader, **options)

    columns = ['ep', 'errC', 'errG', 'err_reconstruction', 'te_acc_C', 'time']
    values = [epoch, train_results['errC'], train_results['errG'], train_results['err_reconstruction'],
              eval_results['classifier_closed_set_accuracy'],
              time.time() - time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if (epoch - 1) % 50 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    # pprint(eval_results)
    if eval_results['classifier_closed_set_accuracy'] > best_acc:
        best_acc = eval_results['classifier_closed_set_accuracy']
        best_epoch = epoch
        save_best_networks(networks, options['result_dir'])

    save_networks(networks, epoch, options['result_dir'])
print("Best Classifier Acc: {} at epoch {}".format(best_acc, best_epoch))
