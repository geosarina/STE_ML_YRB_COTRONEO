import pandas as pd

def read_data(path):
    data = pd.read_csv(path)
    return data

def drop_columns(data, cols_to_drop):
    numerical_data = data.drop(columns=cols_to_drop).astype(float)
    return data, numerical_data

def select_features(data, feature_names):
    reduced_data = data[feature_names]
    return reduced_data
