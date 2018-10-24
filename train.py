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

    for epoch in range(num_epochs):

        logging.info('Epoch {}/{} | LR: {}'.format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr']))

        model.train()
        device = params['device']

        train_summ = []
        train_confusion = np.zeros([2, 2])

        with tqdm(total=len(train_dataloader)) as bar:
            for i, (target, x_cont, x_cat) in enumerate(train_dataloader):
                target, x_cont, x_cat = target.to(device), x_cont.to(device), x_cat.to(device)

                output = model(x_cont, x_cat)
                loss = loss_fn(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluate once in a while
                if i % int(1 / 0.2) == 0:
                    with torch.no_grad():
                        output_batch = output.data.cpu().numpy()
                        labels_batch = target.data.cpu().numpy()

                        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                         for metric in metrics}
                        summary_batch['loss'] = loss.item()
                        train_summ.append(summary_batch)

                        train_confusion += evaluate.confusion(output_batch, labels_batch)

                bar.update()

        metrics_mean = {metric: np.mean([x[metric] for x in train_summ]) for metric in train_summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)

        logging.info("- Train confusion matrix:")
        logging.info(train_confusion)

        valid_summ = []
        valid_confusion = np.zeros([2, 2])

        model.eval()
        with torch.no_grad():
            for i, (target, x_cont, x_cat) in enumerate(val_dataloader):
                target, x_cont, x_cat = target.to(device), x_cont.to(device), x_cat.to(device)

                output = model(x_cont, x_cat)
                loss = loss_fn(output, target)

                output_batch = output.data.cpu().numpy()
                labels_batch = target.data.cpu().numpy()

                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                valid_summ.append(summary_batch)

                valid_confusion += evaluate.confusion(output_batch, labels_batch)

        metrics_mean = {metric: np.mean([x[metric] for x in valid_summ]) for metric in valid_summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)

        logging.info("- Eval confusion matrix:")
        logging.info(valid_confusion)


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
    scheduler = None

    # set loss fn and metrics
    loss_fn = torch.nn.BCELoss()
    metrics = evaluate.metrics

    if click.confirm('Continue with training?', default=False):
        # train model
        logging.info('Starting training for {} epoch(s)'.format(params['max_epochs']))
        try:
            new_model = train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, loss_fn, metrics, params, results_path)
        except KeyboardInterrupt:
            logging.warning('Training interrupted by user.')
            sys.exit(0)
    else:
        logging.warning('Training not started.')
        sys.exit(0)
