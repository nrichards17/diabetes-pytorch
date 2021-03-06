import json
import os
import logging

import pandas as pd
import torch

class Params(dict):
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self, f, indent=4)

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)


class Features(dict):
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self, f, indent=4)

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


def save_metric_histories(train_hist, valid_hist, results_path):
    """
    both lists of dicts, keys are metric names
    """
    train_hist_df = pd.DataFrame(train_hist)
    valid_hist_df = pd.DataFrame(valid_hist)

    TRAIN_HIST_FILE = os.path.join(results_path, 'train_hist.csv')
    VALID_HIST_FILE = os.path.join(results_path, 'valid_hist.csv')

    train_hist_df.to_csv(TRAIN_HIST_FILE, index=False)
    valid_hist_df.to_csv(VALID_HIST_FILE, index=False)


def save_model(model, results_path):
    MODEL_FILE = os.path.join(results_path, 'model.pt')
    torch.save(model, MODEL_FILE)


def load_model(path):
    assert os.path.exists(path), f'File {path} does not exist.'
    return torch.load(path)
