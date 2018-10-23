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


OUTPUT_DATA_DIR = 'data/processed/baseline'

SEED = 42
PCT_THRESHOLD = 95
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
SPLIT_VAR = 'readmitted'

IGNORE_FEATURES = [
    'encounter_id',
    'patient_nbr',
    'weight',
    'payer_code',
    'medical_specialty',
]

CONTINOUS_FEATURES = [
    'age',
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


def drop_rows(df):
    dropped_df = df.copy()

    # drop multiple encounters
    dropped_df = dropped_df.drop_duplicates(subset='patient_nbr', keep='first')
    # drop missing gender
    dropped_df = dropped_df[dropped_df['gender'] != 'Unknown/Invalid']
    # drop missing diag_1
    dropped_df = dropped_df.dropna(subset=['diag_1'])

    expired = [11, 19, 20, 21]

    # drop 11 - Expired
    dropped_df = dropped_df[~dropped_df['discharge_disposition_id'].isin(expired)]

    return dropped_df


def convert_code(code):
    if pd.isnull(code):
        return code

    if ('V' in code) or ('E' in code):
        return 'Other'

    val = int(float(code))

    if (390 <= val <= 459) or (val == 785):
        return 'Circulatory'
    elif (460 <= val <= 519) or (val == 786):
        return 'Respiriatory'
    elif (520 <= val <= 579) or (val == 787):
        return 'Digestive'
    elif (val == 250):
        return 'Diabetes'
    elif (800 <= val <= 999):
        return 'Injury'
    elif (710 <= val <= 739):
        return 'Musculoskeletal'
    elif (580 <= val <= 629) or (val == 788):
        return 'Genitourinary'
    elif (140 <= val <= 239):
        return 'Neoplasms'
    else:
        return 'Other'


def recode_diagnoses(df):
    recoded_df = df.copy()

    recoded_df['diag_1'] = recoded_df['diag_1'].apply(convert_code)
    recoded_df['diag_2'] = recoded_df['diag_2'].apply(convert_code)
    recoded_df['diag_3'] = recoded_df['diag_3'].apply(convert_code)

    return recoded_df


def recode_admission_discharge(df):
    recoded_df = df.copy()

    id_vars = {
        'admission_type_id': [
            [1, 2, 7],  # emergency
            [5, 6, 8]   # null
        ],
        'discharge_disposition_id': [
            [18, 25, 26],   # null
            [11, 19, 20, 21],   # expired
        ],
        'admission_source_id': [
            [9, 15, 17, 20, 21]     # null
        ]
    }
    for key, groups in id_vars.items():
        for group in groups:
            recoded_df[key] = recoded_df[key].replace({x: group[0] for x in group})

    return recoded_df


def age_to_numeric(df):
    ages_df = df.copy()

    ages = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95,
    }

    ages_df['age'] = ages_df['age'].replace(ages)

    return ages_df


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


def identify_imbalanced_categories(df, categorical_features, pct_threshold):
    imbalanced = []
    for var in categorical_features:
        counts = list(df[var].value_counts(dropna=False, normalize=True))
        if counts[0] * 100. > pct_threshold:
            imbalanced.append(var)

    return imbalanced


def encode_data(df, categorical_features):
    encoded_df = df.copy()
    encoded_df['readmitted'] = encoded_df['readmitted'].replace({'>30': 0, '<30': 1})

    for cat_col in categorical_features:
        encoded_df[cat_col] = LabelEncoder().fit_transform(encoded_df[cat_col].astype(str))

    return encoded_df


def get_embedding_dimensions(df, categorical_features):
    cat_dims = [int(df[cat_col].nunique()) for cat_col in categorical_features]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    return emb_dims


def split_data(df, test_size, split_var):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)

    train_iloc, test_iloc = next(splitter.split(df, df[split_var]))
    train_df = df.iloc[train_iloc, :]
    test_df = df.iloc[test_iloc, :]

    return train_df, test_df


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isfile(args.raw_data_path), "Couldn't find the dataset at {}".format(args.raw_data_path)

    print('Reading csv from: {}'.format(args.raw_data_path))
    raw_df = pd.read_csv(args.raw_data_path, na_values='?', low_memory=False)

    print('Starting preprocessing:')

    # replace NO with >30
    print(' - Recasting output: readmitted')
    df = raw_df.copy()
    df['readmitted'] = df['readmitted'].replace({'NO': '>30'})

    print(' - Dropping multiple encounters, missing gender & diag_1, expired')
    before = df.shape
    df = drop_rows(df)
    after = df.shape
    print(f'\t{before} -> {after}')

    print(' - Recoding diagnoses')
    df = recode_diagnoses(df)

    print(' - Recoding admission and discharge ids')
    df = recode_admission_discharge(df)

    print(' - Casting age as numeric')
    df = age_to_numeric(df)

    # downsample majority class in Readmitted output feature
    print(' - Downsampling data')
    before = df.shape
    df = downsample_data(df,
                         feature='readmitted',
                         minority_class='<30',
                         majority_class='>30')
    after = df.shape
    print(f'\t{before} -> {after}')

    categorical_features_all = [var for var in raw_df.columns
                                if var not in IGNORE_FEATURES
                                and var not in CONTINOUS_FEATURES
                                and var not in OUTPUT_FEATURES]

    print(' - Identifying imbalanced categorical features')
    imbalanced_features = identify_imbalanced_categories(df, categorical_features_all, PCT_THRESHOLD)

    # remove imbalanced features from categorical vars
    CATEGORICAL_FEATURES = [var for var in categorical_features_all
                            if var not in imbalanced_features]

    # keep only relevant columns, reorder also
    print(' - Removing unused features')
    before = df.shape
    removed_df = df[OUTPUT_FEATURES + CONTINOUS_FEATURES + CATEGORICAL_FEATURES]
    after = removed_df.shape
    print(f'\t{before} -> {after}')

    # encode categorical variables
    encoded_df = encode_data(removed_df, CATEGORICAL_FEATURES)

    # get embedding dimensions, save as part of features
    embedding_sizes = get_embedding_dimensions(encoded_df, CATEGORICAL_FEATURES)

    # train/val/test split
    n_rows = len(encoded_df)
    test_size = int(TEST_SPLIT * n_rows)
    val_size = int(VAL_SPLIT * n_rows)

    train_df, test_df = split_data(encoded_df, test_size=test_size, split_var=SPLIT_VAR)
    train_df, val_df = split_data(train_df, test_size=val_size, split_var=SPLIT_VAR)

    print(' - Splitting data:')
    print('\tTrain: {}'.format(train_df.shape))
    print('\tVal: {}'.format(val_df.shape))
    print('\tTest: {}'.format(test_df.shape))

    if not os.path.exists(OUTPUT_DATA_DIR):
        print('Creating: output dir {}'.format(OUTPUT_DATA_DIR))
        os.mkdir(OUTPUT_DATA_DIR)
    else:
        print("Warning: output dir {} already exists".format(OUTPUT_DATA_DIR))

    TRAIN_FILE = os.path.join(OUTPUT_DATA_DIR, 'train.csv')
    VAL_FILE = os.path.join(OUTPUT_DATA_DIR, 'val.csv')
    TEST_FILE = os.path.join(OUTPUT_DATA_DIR, 'test.csv')

    print(f'Saving train_df to:\n  {TRAIN_FILE}')
    train_df.to_csv(TRAIN_FILE, index=False)
    print(f'Saving val_df to:\n  {VAL_FILE}')
    val_df.to_csv(VAL_FILE, index=False)
    print(f'Saving test_df to:\n  {TEST_FILE}')
    test_df.to_csv(TEST_FILE, index=False)

    variables = {
        'continuous': CONTINOUS_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'output': OUTPUT_FEATURES
    }

    features = utils.Features()
    features.update(variables)
    features['embedding_sizes'] = embedding_sizes

    FEATURE_FILE = os.path.join(OUTPUT_DATA_DIR, 'features.json')

    print(f'Saving features to:\n  {FEATURE_FILE}')
    features.save(FEATURE_FILE)

    print('Done.')
