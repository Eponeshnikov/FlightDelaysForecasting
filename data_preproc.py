import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def fit_encoders(data):
    """
    Function which returns fitted encoders
    :param data: input dataframe
    :return: onehotencoder, label encoder
    """
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    label_enc = LabelEncoder()
    cat_feats = ['Depature Airport', 'Destination Airport']
    airports = list(set(pd.concat([data[cat_feats[0]], data[cat_feats[1]]]).reset_index(drop=True)))
    ohe_encoder.fit(data[cat_feats])
    label_enc.fit(airports)
    return ohe_encoder, label_enc


def ohe_new_features(data, features_name, encoder):
    """
    Apply onehotencoder to dataframe
    :param data: input dataframe
    :param features_name: list of feature names
    :param encoder: encoder object
    :return: modified dataframe
    """
    new_feats = encoder.transform(data[features_name])
    new_cols = pd.DataFrame(new_feats, dtype=int)
    new_df = pd.concat([data, new_cols], axis=1)
    new_df.drop(features_name, axis=1, inplace=True)
    return new_df


def add_features(data, *params):
    """
    Add new features
    :param data: input dataframe
    :param params: features to add
    :return: modified dataframe
    """
    if 'flight_dur' in params:
        data['Flight Duration'] = pd.to_datetime(data["Scheduled arrival time"]) \
                                  - pd.to_datetime(data["Scheduled depature time"])
        data['Flight Duration'] = data['Flight Duration'].dt.total_seconds() / 60
    if 'year' in params:
        data['year'] = pd.to_datetime(data["Scheduled depature time"]).dt.year
    if 'month' in params:
        data['month'] = pd.to_datetime(data["Scheduled depature time"]).dt.month
    if 'day' in params:
        data['day'] = pd.to_datetime(data["Scheduled depature time"]).dt.day
    return data


def drop_columns(data, *columns, reset=True):
    """
    Drop columns
    :param data: input dataframe
    :param columns: columns to drop
    :param reset: reset indexes
    :return: modified dataframe
    """
    for column in columns:
        data = data.drop(column, axis=1).reset_index(drop=reset)
    return data


def le_new_features(data, features_names, encoder):
    """
    Apply label encoder to dataframe
    :param features_names: list of feature names
    :param data: input dataframe
    :param encoder: encoder object
    :return: modified dataframe
    """
    for features_name in features_names:
        data[features_name] = encoder.transform(data[features_name])
    return data


def train_test_spl(data, year=2018):
    """
    Split dataframe to train and test
    :param data: input dataframe
    :param year: year as point to split
    :return: train and test dataframes
    """
    train = data.where(pd.to_datetime(data["Scheduled depature time"]).dt.year < year)
    train = train[train["Scheduled depature time"].notna()]
    train = train.drop(["Scheduled depature time", "Scheduled arrival time"], axis=1).reset_index(drop=True)
    test = data.where(pd.to_datetime(data["Scheduled depature time"]).dt.year >= year)
    test = test[test["Scheduled depature time"].notna()]
    test = test.drop(["Scheduled depature time", "Scheduled arrival time"], axis=1).reset_index(drop=True)
    return train, test


def x_y_spl(data):
    """
    Split dataframe to predictors and labels
    :param data: input dataframe
    :return: predictors, labels
    """
    y = data["Delay"]
    x = data.drop("Delay", axis=1).reset_index(drop=True)
    return x, y


def remove_outliers(data, thresh):
    """
    Remove outliers using z-test
    :param data: input dataframe
    :param thresh: threshold for z-test
    :return: modified dataframe
    """
    z = np.abs(stats.zscore(data))
    # return data[(z < thresh)['Delay'] == (z < thresh)['flight_duration']]
    return data[(z < thresh).all(axis=1)].reset_index(drop=True)
