import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy(output, target, threshold=0.5):
    output[output >= threshold] = 1
    output[output < threshold] = 0
    return accuracy_score(target, output)


def auroc(output, target):
    return roc_auc_score(target, output)


def evaluate(model, loss_fn, dataloader, metrics, params):
    """
    todo
    """

    model.eval()

    # device: gpu or cpu
    device = params['device']

    summ = []

    with torch.no_grad():
        for i, (target, X_cont, X_cat) in enumerate(dataloader):
            target, X_cont, X_cat = target.to(device), X_cont.to(device), X_cat.to(device)

            output = model(X_cont, X_cat)
            loss = loss_fn(output, target)

            output_batch = output.data.cpu().numpy()
            labels_batch = target.data.cpu().numpy()

            # print(output_batch.ravel())
            # print(labels_batch.ravel())
            # print()

            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


metrics = {
    'accuracy': accuracy,
    'auroc': auroc,
}
