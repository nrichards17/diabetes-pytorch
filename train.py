import argparse
import logging
import os
import sys
from tqdm import tqdm
import click

import numpy as np
import torch
from torch.optim import Adam

import utils
import model.data_loader as data_loader
import model.net as net
import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', help='Path to experiment json file.')
parser.add_argument('--path_to_results', default='results/', help='Path to results directory.')
parser.add_argument('--path_to_data', default='data/processed', help='Path to processed data directory.')

SEED = 42


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn,
                       metrics, params, results_path):

    num_epochs = params['max_epochs']
    train_histories, valid_histories = [], []
    best_valid_auroc = 0.0

    for epoch in range(num_epochs):

        logging.info(' - Epoch {}/{} | LR: {}'.format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr']))

        model.train()
        device = params['device']

        with tqdm(total=len(train_dataloader)) as bar:
            for i, (target, x_cont, x_cat) in enumerate(train_dataloader):
                target, x_cont, x_cat = target.to(device), x_cont.to(device), x_cat.to(device)

                output = model(x_cont, x_cat)
                loss = loss_fn(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bar.update()

        train_metrics, train_confusion = evaluate.evaluate(model, loss_fn, train_dataloader, metrics, params)
        train_histories.append(train_metrics)

        train_metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
        logging.info(" - Train metrics: " + train_metrics_string)
        logging.info(" - Train confusion matrix: \n{}".format(train_confusion))

        # evaluate on validation set
        valid_metrics, valid_confusion = evaluate.evaluate(model, loss_fn, val_dataloader, metrics, params)
        valid_histories.append(valid_metrics)

        valid_metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in valid_metrics.items())
        logging.info(" - Val metrics : " + valid_metrics_string)
        logging.info(" - Valid confusion matrix: \n{}".format(valid_confusion))

        curr_valid_auroc = valid_metrics['auroc']
        if curr_valid_auroc >= best_valid_auroc:
            logging.info(" - New best validation AUROC - saving model.")
            best_valid_auroc = curr_valid_auroc
            utils.save_model(model, results_path)

        # save histories to csv
        utils.save_metric_histories(train_histories, valid_histories, results_path)


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
    logging.info(model)

    # optimizer, scheduler = create_opt_from_params(model, params)
    optimizer = Adam(model.parameters(), lr=params['learning_rate'])

    # set loss fn and metrics
    loss_fn = torch.nn.BCELoss()
    metrics = evaluate.metrics

    if click.confirm('Continue with training?', default=False):
        # train model
        logging.info('Starting training for {} epoch(s)'.format(params['max_epochs']))
        try:
            new_model = train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, results_path)
        except KeyboardInterrupt:
            logging.warning('Training interrupted by user.')
            sys.exit(0)
    else:
        logging.warning('Training not started.')
        sys.exit(0)
