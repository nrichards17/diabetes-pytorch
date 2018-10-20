import argparse
import logging
import os

import numpy as np
import torch

import utils

parser = argparse.ArgumentParser()
# todo


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    pass


if __name__ == '__main__':
    # Load json params

    # set device
    params.device = torch.device('gpu:0')

    # set random seeds

    # set logger

    # fetch dataloaders

    # create model and optimizer

    # set loss fn and metrics

    # train model
