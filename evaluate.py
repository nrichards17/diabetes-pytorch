import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def threshold_output(output, threshold=0.5):
    t_output = np.zeros_like(output)
    t_output[output >= threshold] = 1.0
    t_output[output < threshold] = 0.0

    return t_output


def accuracy(output, target, threshold=0.5):
    t_output = threshold_output(output, threshold=threshold)

    return accuracy_score(target, t_output)


def auroc(output, target):
    return roc_auc_score(target, output)


def confusion(output, target):
    t_output = threshold_output(output)

    return confusion_matrix(target, t_output)


def evaluate(model, loss_fn, dataloader, metrics, params):
    """
    todo
    """
    device = params['device']

    summaries = []
    matrix = np.zeros([2, 2])

    model.eval()
    with torch.no_grad():
        for i, (target, x_cont, x_cat) in enumerate(dataloader):
            target, x_cont, x_cat = target.to(device), x_cont.to(device), x_cat.to(device)

            output = model(x_cont, x_cat)
            loss = loss_fn(output, target)

            output_batch = output.data.cpu().numpy()
            labels_batch = target.data.cpu().numpy()

            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summaries.append(summary_batch)

            matrix += confusion(output_batch, labels_batch)

    metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}

    return metrics_mean, matrix


metrics = {
    'accuracy': accuracy,
    'auroc': auroc,
}
