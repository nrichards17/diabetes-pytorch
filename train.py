import argparse
import logging
import os

import numpy as np
import torch

import utils
import model.data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', help='Path to experiment json file.')
parser.add_argument('--path_to_data', default='data/processed', help='Path to processed data directory.')

SEED = 42


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    pass


if __name__ == '__main__':
    args = parser.parse_args()

    # load json params
    params_path = args.experiment
    assert os.path.isfile(params_path), "No json configuration file found at {}".format(params_path)

    params = utils.Params()
    params.load(params_path)

    # load json features
    features_path = os.path.join(args.path_to_data, params['dataset'], 'features.json')
    assert os.path.isfile(features_path), "No json features file found at {}".format(features_path)

    features = utils.Features()
    features.load(features_path)

    # params.update(features)
    # print(params)

    # set device
    params['cuda'] = torch.cuda.is_available()
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # set random seeds
    torch.manual_seed(SEED)
    if params['cuda']: torch.cuda.manual_seed(SEED)

    # set logger

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloaders(data_dir=args.path_to_data, features=features, params=params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    # create model and optimizer

    # set loss fn and metrics

    # train model
