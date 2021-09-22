from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
import numpy as np
import pandas as pd


def fit_poly_reg(x_train, y_train, x_test, y_test, regressions, degrees):
    pipelines = dict()
    scores = dict()
    for regression in regressions:
        for i in range(len(degrees)):
            score_mod = dict()
            if degrees[i] != 1:
                print(f"Fit {degrees[i]} degree of {str(regression)}...")
            else:
                print(f"Fit {str(regression)}...")
            polynomial_features = PolynomialFeatures(degree=degrees[i])
            regression_ = regression
            pipeline = Pipeline([("polynomial_features", polynomial_features),
                                 ("regression", regression_)])
            pipeline = pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            score_mod['mean_squared_error'] = mse
            score_mod['mean_absolute_error'] = mae
            pipelines[f"{str(regression)}_{degrees[i]}"] = pipeline
            scores[f"{str(regression)}_{degrees[i]}"] = score_mod
            print('')
        print('-' * 30)
    return pipelines, scores


def cross_val_test(x_train, y_train, models, errors, cv=5):
    scores = dict()
    for model in models:
        scores_mod = dict()
        print(f"Evaluating {model}:")
        for error in errors:
            print(f"Calculating {error}...")
            print(x_train.shape)
            score = cross_val_score(models[model], x_train, y_train,
                                    scoring=error, cv=cv)
            scores_mod[error[4:]] = score
            print('')
        scores[model] = scores_mod
        print('-' * 30)
    return scores


def test_models(x_test, y_test, models, errors):
    scores = dict()
    for model in models:
        scores_mod = dict()
        print(f"Testing {model}:")
        y_pred = models[model].predict(x_test)
        for error in errors:
            print(f"Calculating {error}...")
            score = 0
            if error == 'neg_mean_absolute_error':
                score = metrics.mean_absolute_error(y_test, y_pred)
            elif error == 'neg_mean_squared_error':
                score = metrics.mean_squared_error(y_test, y_pred)
            scores_mod[error[4:]] = score
            print('')
        scores[model] = scores_mod
        print('-'*30)
    return scores

