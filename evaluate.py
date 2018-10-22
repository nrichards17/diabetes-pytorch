import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy(output, target, threshold=0.5):
    output[output >= threshold] = 1
    output[output < threshold] = 0
    return accuracy_score(target, output)


def auroc(output, target):
    return roc_auc_score(target, output)


metrics = {
    'accuracy': accuracy,
    'auroc': auroc,
}
