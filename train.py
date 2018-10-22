import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import Adam

from richml.scheduler import OneCycleLR

import utils
import model.data_loader as data_loader
import model.net as net


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', help='Path to experiment json file.')
parser.add_argument('--path_to_results', default='results/', help='Path to results directory.')
parser.add_argument('--path_to_data', default='data/processed', help='Path to processed data directory.')

SEED = 42


def create_opt_from_params(model, params):
    optim_params = params['optimizer']
    sched_params = params['scheduler']

    if optim_params['type'] == 'Adam':
        optimizer = Adam(model.parameters(), weight_decay=optim_params['weight_decay'])
    else:
        raise NotImplementedError('Optimizer {} not implemented'.format(optim_params['type']))

    if sched_params['type'] == 'OneCycle':
        scheduler = OneCycleLR(
            optimizer,
            max_epochs=sched_params['max_epochs'],
            eta_min=sched_params['eta_min'],
            eta_max=sched_params['eta_max'],
            epsilon=sched_params['epsilon'],
            end_fraction=sched_params['end_fraction']
        )
    else:
        raise NotImplementedError('Scheduler {} not implemented'.format(sched_params['type']))

    return optimizer, scheduler


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """
    todo
    """

    # set model to training mode
    model.train()

    # device: gpu or cpu
    device = params['device']

    with tqdm(total=len(dataloader)) as bar:
        for i, (target, X_cont, X_cat) in enumerate(dataloader):
            target, X_cont, X_cat = target.to(device), X_cont.to(device), X_cat.to(device)

            output = model(X_cont, X_cat)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            bar.update()

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn,
                       metrics, params, results_path):

    num_epochs = params['scheduler']['max_epochs']

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # step scheduler

        # evaluate


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
    # update params with features - needed for network construction
    params.update(features)

    # create results directory
    results_path = os.path.join(args.path_to_results, params['name'])
    if not os.path.exists(results_path):
        print('Creating results dir: {}'.format(results_path))
        os.mkdir(results_path)
    else:
        print('Warning: results dir {} already exists'.format(results_path))

    # set device
    params['cuda'] = torch.cuda.is_available()
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seeds
    torch.manual_seed(SEED)
    if params['cuda']: torch.cuda.manual_seed(SEED)

    # set logger
    utils.set_logger(os.path.join(results_path, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloaders(data_dir=args.path_to_data, features=features, params=params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # create model and optimizer
    model = net.Network(params)
    model.to(params['device'])

    optimizer, scheduler = create_opt_from_params(model, params)

    # set loss fn and metrics
    loss_fn = torch.nn.BCELoss()
    # metrics
    metrics = {}

    # train model
    logging.info('Starting training for {} epoch(s)'.format(params['scheduler']['max_epochs']))
    train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, loss_fn, metrics, params, results_path)