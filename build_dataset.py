import argparse
import os

import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 999
pd.options.display.max_rows = 20


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', default='data/raw/diabetic_data.csv', help="Directory with the raw diabetes dataset")
parser.add_argument('--output_dir', default='data/processed/baseline', help="Where to write the new data")

SEED = 42
PCT_THRESHOLD = 95
TEST_SPLIT = 0.2
SPLIT_VAR = 'readmitted'

IGNORE_FEATURES = [
    'encounter_id',
    'patient_nbr',
]

KEEP_FEATURES = [
    'weight',
]

CONTINOUS_FEATURES = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
]

OUTPUT_FEATURES = ['readmitted']


def identify_imbalanced_categories(df, categorical_features, pct_threshold):
    imbalanced = []
    for var in categorical_features:
        counts = list(df[var].value_counts(dropna=False, normalize=True))
        if counts[0] * 100. > pct_threshold:
            imbalanced.append(var)

    return imbalanced


def downsample_data(df, feature, minority_class, majority_class):
    # number of cases readmitted <30 - minority class
    num_minor_samples = df[feature].value_counts()[minority_class]

    minority_df = df[df[feature] == minority_class]
    majority_df = df[df[feature] == majority_class]

    # downsample majority class to same as minority class
    majority_downsampled_df = majority_df.sample(num_minor_samples, replace=False, random_state=SEED)

    # create new df w/ 50/50 classes
    downsampled_df = pd.concat([minority_df, majority_downsampled_df])
    # shuffle new df and reset index
    downsampled_df = downsampled_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return downsampled_df


def encode_data(df, continuous_features, categorical_features, output_features):
    encoded_df = df[output_features + continuous_features + categorical_features]
    encoded_df['readmitted'] = encoded_df['readmitted'].replace({'>30': 0, '<30': 1})

    for cat_col in categorical_features:
        encoded_df[cat_col] = LabelEncoder().fit_transform(encoded_df[cat_col].astype(str))

    return encoded_df


def split_data(df, split_var):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)

    train_iloc, test_iloc = next(splitter.split(df, df[split_var]))
    train_df = df.iloc[train_iloc, :]
    test_df = df.iloc[test_iloc, :]

    return train_df, test_df


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isfile(args.raw_data_path), "Couldn't find the dataset at {}".format(args.raw_data_path)

    print('Reading csv from: {}'.format(args.raw_data_path))
    raw_df = pd.read_csv(args.raw_data_path, na_values='?', low_memory=False)

    # replace NO with >30
    raw_df['readmitted'] = raw_df['readmitted'].replace({'NO': '>30'})

    # downsample majority class in Readmitted output feature
    downsampled_df = downsample_data(raw_df,
                                     feature='readmitted',
                                     minority_class='<30',
                                     majority_class='>30')

    categorical_features_all = [var for var in raw_df.columns
                                if var not in IGNORE_FEATURES
                                and var not in CONTINOUS_FEATURES
                                and var not in OUTPUT_FEATURES]

    imbalanced_features = identify_imbalanced_categories(downsampled_df, categorical_features_all, PCT_THRESHOLD)

    # remove imbalanced features from categorical vars
    CATEGORICAL_FEATURES = [var for var in categorical_features_all
                            if var not in imbalanced_features
                            or var in KEEP_FEATURES]

    # encode categorical variables
    encoded_df = encode_data(downsampled_df, CONTINOUS_FEATURES, CATEGORICAL_FEATURES, OUTPUT_FEATURES)

    # train/test split
    train_df, test_df = split_data(encoded_df, split_var=SPLIT_VAR)

    if not os.path.exists(args.output_dir):
        print('Creating: output dir {}'.format(args.output_dir))
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    TRAIN_FILE = os.path.join(args.output_dir, 'train.csv')
    TEST_FILE = os.path.join(args.output_dir, 'test.csv')

    print(f'Saving train_df to:\n  {TRAIN_FILE}')
    train_df.to_csv(TRAIN_FILE, index=False)
    print(f'Saving test_df to:\n  {TEST_FILE}')
    test_df.to_csv(TEST_FILE, index=False)

    variables = {
        'continuous': CONTINOUS_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'output': OUTPUT_FEATURES
    }

    features = utils.Features()
    features.update(variables)

    FEATURE_FILE = os.path.join(args.output_dir, 'features.json')

    print(f'Saving features to:\n  {FEATURE_FILE}')
    features.save(FEATURE_FILE)
