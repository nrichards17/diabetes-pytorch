import argparse
import os

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', default='data/raw/diabetic_data.csv', help="Directory with the raw diabetes dataset")
parser.add_argument('--output_dir', default='data/processed', help="Where to write the new data")

SEED = 42
PCT_THRESHOLD = 95

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


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isfile(args.raw_data_path), "Couldn't find the dataset at {}".format(args.raw_data_path)

    print('Reading csv from: {}'.format(args.raw_data_path))
    raw_df = pd.read_csv(args.raw_data_path, na_values='?', low_memory=False)

    # replace NO with >30
    raw_df['readmitted'] = raw_df['readmitted'].replace({'NO': '>30'})

    downsampled_df = downsample_data(raw_df,
                                     feature='readmitted',
                                     minority_class='<30',
                                     majority_class='>30')

    categorical_features_all = [var for var in raw_df.columns
                                if var not in IGNORE_FEATURES
                                and var not in CONTINOUS_FEATURES
                                and var not in OUTPUT_FEATURES]

    imbalanced_features = identify_imbalanced_categories(downsampled_df, categorical_features_all, PCT_THRESHOLD)

    CATEGORICAL_FEATURES = [var for var in categorical_features_all
                            if var not in imbalanced_features
                            or var in KEEP_FEATURES]

    # print(categorical_features_all)
    # print(imbalanced_features)
    # print(CATEGORICAL_FEATURES)

    embedded_df = downsampled_df[OUTPUT_FEATURES + CONTINOUS_FEATURES + CATEGORICAL_FEATURES]
    embedded_df['readmitted'] = embedded_df['readmitted'].replace({'>30': 0, '<30': 1})

    for cat_col in CATEGORICAL_FEATURES:
        print(cat_col)
        embedded_df[cat_col] = LabelEncoder().fit_transform(embedded_df[cat_col].astype(str))

    print(embedded_df)
    print(embedded_df.dtypes)

