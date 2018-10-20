import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', default='data/raw/diabetic_data.csv', help="Directory with the raw diabetes dataset")
parser.add_argument('--output_dir', default='data/processed', help="Where to write the new data")

SEED = 42
PCT_THRESHOLD = 95

ignore_features = [
    'encounter_id',
    'patient_nbr',
]

continuous_features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
]

output_features = ['readmitted']


def identify_imbalanced_categories(df, categorical_features, pct_threshold):
    imbalanced = []
    for var in categorical_features:
        counts = list(df[var].value_counts(dropna=False, normalize=True))
        if counts[0] * 100. > pct_threshold:
            imbalanced.append(var)

    return imbalanced


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isfile(args.raw_data_path), "Couldn't find the dataset at {}".format(args.raw_data_path)

    print('Reading csv from: {}'.format(args.raw_data_path))
    raw_df = pd.read_csv(args.raw_data_path, na_values='?')
    print('Replacing "NO" with ">30"')
    raw_df['readmitted'] = raw_df['readmitted'].replace({'NO': '>30'})

    # number of cases readmitted <30 - minority class
    num_minor_samples = raw_df['readmitted'].value_counts()['<30']

    minority_df = raw_df[raw_df['readmitted'] == '<30']
    majority_df = raw_df[raw_df['readmitted'] == '>30']

    # downsample majority class to same as minority class
    majority_downsampled_df = majority_df.sample(num_minor_samples,
                                                 replace=False,
                                                 random_state=SEED)

    # create new df w/ 50/50 classes
    downsampled_df = pd.concat([minority_df, majority_downsampled_df])
    # shuffle new df and reset index
    downsampled_df = downsampled_df.sample(frac=1, random_state=SEED).reset_index(drop=True)





    categorical_features = [var for var in raw_df.columns
                            if var not in ignore_features
                            and var not in continuous_features
                            and var not in output_features]

    imbalanced_features = identify_imbalanced_categories(raw_df, categorical_features, PCT_THRESHOLD)

