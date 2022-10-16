import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def separate_inputs_and_labels(data):
    y = data['d7Li']
    X = data.drop(['d7Li'], axis=1)
    return X, y

def train_model(model, data, seed, scale=True):

    X, y = separate_inputs_and_labels(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)

    model.fit(X_train, y_train)

    return model, X, X_train, X_test, y_test

def evaluate_model(model, X, X_train, X_test, y_test, seed):
    eval_params = {}

    importances = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=seed).importances_mean

    y_pred = model.predict(X_test)

    r_square = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mae_mean = mean_absolute_error(y_test, np.repeat(np.mean(y_test), len(y_test)))
    residuals = y_test - y_pred

    indices = np.argsort(importances)

    eval_params['r_square'] = r_square
    eval_params['mae'] = mae
    eval_params['mae_mean'] = mae_mean
    eval_params['importance_vals'] = importances[indices]
    eval_params['importance_names'] = X.columns[indices]
    eval_params['residuals'] = residuals
    eval_params['final_model'] = model
    eval_params['y_pred'] = y_pred
    eval_params['y_test'] = y_test
    eval_params['X_train'] = X_train

    return eval_params

def iterated_evaluation(model, data, model_name, num_iterations=10):
    all_evals = {
    'r_square': [],
    'mae': [],
    'mae_mean': [],
    'importance_vals': [],
    'importance_names': [],
    'residuals': [],
    'final_model': [],
    'y_pred': [],
    'y_test': [],
    'X_train': [],
    }

    scale = True

    if model_name == 'model2':
        scale = False

    for seed in np.arange(num_iterations):
        trained_model, X, X_train, X_test, y_test = train_model(model, data, seed, scale=True)
        eval_params = evaluate_model(trained_model, X, X_train, X_test, y_test, seed)

        all_evals['r_square'].append(eval_params['r_square'])
        all_evals['mae'].append(eval_params['mae'])
        all_evals['mae_mean'].append(eval_params['mae_mean'])
        all_evals['importance_vals'].append(eval_params['importance_vals'])
        all_evals['importance_names'].append(eval_params['importance_names'])
        all_evals['residuals'].append(eval_params['residuals'])
        all_evals['final_model'].append(eval_params['final_model'])
        all_evals['y_pred'].append(eval_params['y_pred'])
        all_evals['y_test'].append(eval_params['y_test'])
        all_evals['X_train'].append(eval_params['X_train'])

    return all_evals

def save_plots_for_all_models(model_evaluations, path):
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_axes([0,0,1,1])

    #Note: the order in which these models are plotted came from trial and
    #error. In our case, it happens to be the case that model 2 (RF) performs best,
    #and model 1 performs worst.
    sns.violinplot(data=[
    model_evaluations['model1']['mae_mean'],
    model_evaluations['model1']['mae'],
    model_evaluations['model4']['mae'],
    model_evaluations['model3']['mae'],
    model_evaluations['model2']['mae']],
    palette=['r','b', 'm', 'y', 'g'],
    saturation=0.5
    )

    plt.ylim(0)

    plt.xticks(np.arange(5), ['mean', 'lin reg', 'knn', 'svm', 'rf'])
    plt.xticks(rotation=90)
    plt.title('Model comparisons - MAE')
    plt.savefig(path, dpi=150)
    plt.clf()

def test_all_models(models_to_test, data):
    model_evaluations = {}
    for model in models_to_test:
        model_evaluations[model] = iterated_evaluation(models_to_test[model], data, model, num_iterations=30)
        print('Completed evaluation for ' + model)

    return model_evaluations

def locate_median_model(model_evaluations):
    median_performance = np.median(model_evaluations['mae'][1:])
    median_model_index = model_evaluations['mae'].index(median_performance)
    return median_model_index

def retrieve_evals_for_model_at_index(model_evaluations, index):
    median_evaluations = {}

    median_evaluations['r_square'] = model_evaluations['r_square'][index]
    median_evaluations['mae'] = model_evaluations['mae'][index]
    median_evaluations['mae_mean'] = model_evaluations['mae_mean'][index]
    median_evaluations['importance_vals'] = model_evaluations['importance_vals'][index]
    median_evaluations['importances_names'] = model_evaluations['importance_names'][index]
    median_evaluations['residuals'] = model_evaluations['residuals'][index]
    median_evaluations['y_pred'] = model_evaluations['y_pred'][index]
    median_evaluations['y_test'] = model_evaluations['y_test'][index]
    median_evaluations['model'] = model_evaluations['final_model'][index]
    median_evaluations['X_train'] = model_evaluations['X_train'][index]

    return median_evaluations

def retrieve_median_model_evals(model_evaluations):
    median_model_index = locate_median_model(model_evaluations)
    median_model_evaluations = retrieve_evals_for_model_at_index(model_evaluations, median_model_index)
    return median_model_evaluations

def save_y_test_histogram(y_test, path):
    y_test.hist()
    plt.title('y_test histogram')
    plt.savefig(path, dpi=150)
    plt.clf()

def save_residuals_histogram(residuals, path):
    residuals.hist()
    plt.title('residuals')
    plt.savefig(path, dpi=150)
    plt.clf()

def save_y_pred_vs_y_test(y_pred, y_test, path):
    fig = plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred)
    plt.title('y_pred vs y_test')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.savefig(path, dpi=150)
    plt.clf()

def save_residuals_vs_y_test(residuals, y_test, path):
    fig = plt.figure(figsize=(8,6))
    plt.scatter(y_test, residuals)
    plt.title('residuals vs y_test')
    plt.xlabel('y_test')
    plt.ylabel('residuals')
    plt.savefig(path, dpi=150)
    plt.clf()

def save_feature_importance_plots(importance_names, importance_vals, path):
    plt.figure(figsize=(12,4))
    plt.title("Feature importances")
    plt.bar(range(len(importance_names)), importance_vals,
        color="r", align="center")
    plt.xticks(range(len(importance_names)), importance_names)
    plt.xlim([-1, len(importance_names)])
    plt.xticks(rotation=90)
    plt.savefig(path, dpi=150)
    plt.clf()

def save_partial_dependence_plot(model, X_train, columns, path):
    pdp_plots = plot_partial_dependence(model, X_train, columns,
        kind="average", random_state=0)
    pdp_plots.figure_.suptitle('Partial dependence of Li isotope ratio on most important features')
    pdp_plots.figure_.subplots_adjust(hspace=0.3)
    pdp_plots.figure_.set_size_inches((16,12))
    plt.savefig(path, dpi=150)
    plt.clf()
