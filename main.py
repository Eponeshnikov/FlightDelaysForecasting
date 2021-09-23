import pandas as pd
import numpy as np
import contextlib
import io
import data_preproc
import visualisation
import fit_test
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# Resetting restrictions on the number of displayed rows, columns, records
def pandas_options(max_rows=True, max_columns=True, max_colwidth=False):
    """
    Resetting restrictions on the number of displayed rows, columns, records
    :param max_rows: reset rows restrictions
    :param max_columns: reset columns restrictions
    :param max_colwidth: reset column width restrictions
    :return:
    """
    if max_rows:
        pd.set_option('display.max_rows', None)
    if max_columns:
        pd.set_option('display.max_columns', None)
    if max_colwidth:
        pd.set_option('display.max_colwidth', None)


def prepare_data(return_data=False):
    """
    Returns data with new features ('Flight duration', 'year', 'month', 'day');
    drop columns 'Scheduled depature time', 'Scheduled arrival time';
    remove outliers with threshold=3
    split data to train and test
    :param return_data: not split data
    :return:
    """
    raw_data = pd.read_csv("flight_delay.csv")
    cat_feats = ['Depature Airport', 'Destination Airport']
    ohe_enc, label_enc = data_preproc.fit_encoders(raw_data)
    data = data_preproc.add_features(raw_data, 'flight_dur', 'year', 'month', 'day')
    data = data_preproc.le_new_features(data, cat_feats, label_enc)
    train, test = data_preproc.train_test_spl(data)
    data = data_preproc.drop_columns(data, 'Scheduled depature time', 'Scheduled arrival time')
    train = data_preproc.remove_outliers(train, 3)
    x_train, y_train = data_preproc.x_y_spl(train.copy())
    x_test, y_test = data_preproc.x_y_spl(test.copy())
    if return_data:
        return data
    else:
        return x_train, x_test, y_train, y_test


def generate_imgs(data):
    """
    Generates figures with different thresholds
    :param data: input dataframe
    :return: None
    """
    max_cmap = max(data['Delay'])
    for i in range(1, 5):
        data_vis = data_preproc.remove_outliers(data.copy(), i)
        visualisation.vis_params_delay(data_vis, max_cmap, 'Flight Duration', 'Depature Airport',
                                       'Destination Airport', 'year', 'month', 'day', name_=i)
    data_vis = data.copy()
    visualisation.vis_params_delay(data_vis, max_cmap, 'Flight Duration', 'Depature Airport',
                                   'Destination Airport', 'year', 'month', 'day', name_="raw")


def fit_models(x_train, y_train, x_test, y_test):
    """
    Generate, fit and test models
    :param x_train: train predictors
    :param y_train: train labels
    :param x_test: test predictors
    :param y_test: test labels
    :return: dict of fitted models, errors score of testing
    """
    alphas = [i / 10 for i in range(1, 20, 2)]  # generate diff alphas for lasso and ridge
    linear_reg = [LinearRegression()]  # linear reg
    ridges = [Ridge(i) for i in alphas]  # list of ridges with diff alphas
    lassos = [Lasso(i) for i in alphas]  # list of lassos with diff alphas
    models = linear_reg + ridges + lassos  # concat all regressions
    degrees = [i for i in range(1, 4)]  # generates degrees
    pipelines, scores = fit_test.fit_poly_reg(x_train, y_train, x_test, y_test, models, degrees)  # fit models
    return pipelines, scores


def eval_models(x_train, y_train, models, errors):
    """
    Evaluates models using cross-validation and prints results
    :param x_train: train predictors
    :param y_train: train labels
    :param models: dict of models
    :param errors: list of errors
    :return: None
    """
    print("Evaluation Errors:")
    scores = fit_test.cross_val_test(x_train, y_train, models, errors)
    for i, score in enumerate(scores):
        for err in scores[score]:
            print(f"{score}:\n{err}: {round(-np.mean(scores[score][err]), 2)} "
                  f"+/- {round(np.std(scores[score][err], ddof=1), 2)}")
    print('=' * 30)


def print_test_results(scores):
    """
    Print errors scores of testing
    :param scores: dict of errors scores
    :return: None
    """
    # scores = fit_test.test_models(x_test, y_test, models, errors)
    print("Test Errors:")
    for i, score in enumerate(scores):
        for err in scores[score]:
            print(f"{score}:\n{err}: {round(scores[score][err], 2)}")
    print('=' * 30)


def txt_all(name_, *params):
    """
    Append arbitrary number of strings into .txt file
    :param name_: first part of filename of .txt file. The second part of filename is log.txt
    :param params: strings to write
    :return: None
    """
    filename = name_ + "log.txt"  # generate filename
    flag = "w"
    # write strings
    with open(filename, flag) as f:
        for p in params:
            f.write(p)
            f.write('\n')


def main():
    """
    Main function: prepare data, generate, fit, evaluate models, generate figures
    :return: None
    """
    pandas_options()
    data = prepare_data(return_data=True)
    x_train, x_test, y_train, y_test = prepare_data()
    errors = ["neg_mean_squared_error", 'neg_mean_absolute_error']
    generate_imgs(data)
    models, test_scores = fit_models(x_train, y_train, x_test, y_test)
    eval_models(x_train, y_train, models, errors)
    print_test_results(test_scores)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()) as f_:
        main()
    # write to txt all information
    out = f_.getvalue()
    txt_all("test", out)
