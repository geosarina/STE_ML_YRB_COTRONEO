from data_preprocessing import *
from data_visualization import *
from modeling_and_evaluation import *
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

seed = 11
figure_resolution = 600

if __name__ == '__main__':

    #Data preprocessing

    data = read_data('../data.csv')
    cols_to_drop = ['Sample_ID']
    data, numerical_data = drop_columns(data, cols_to_drop)

    #Data visualization

    corr = generate_correlation_matrix(numerical_data)

    save_correlation_matrix(corr, "../figures/correlation_matrix", figure_resolution)
    save_correlations_with_d7Li(corr, "../figures/d7Li_correlations", figure_resolution)
    save_scatter_matrix(numerical_data, "../figures/scatter_matrix", figure_resolution)

    #Showing an example of a pair of columns you might want to compare
    columns_to_compare = ["d7Li", "MAT"]
    save_partial_scatter_matrix(numerical_data, columns_to_compare, "../figures/partial_scatter_matrix", figure_resolution)

    #Dimensionality reduction (PCA and t-SNE)

    data_no_Li = remove_Li(numerical_data)
    scaled_data_no_Li = scale_data(data_no_Li)

    n_components = 7
    pca_data, explained_variance_ratios, pca = run_pca(scaled_data_no_Li, n_components)

    save_all_pca_component_contributions(pca, data_no_Li, n_components, "../figures/PC", figure_resolution)
    save_explained_variance_ratios(explained_variance_ratios, "../figures/explained_variance_ratios", figure_resolution)
    save_cumulative_variance(explained_variance_ratios, "../figures/cumulative_variance", figure_resolution)

    pca_data_2d = generate_2_component_pca_df(pca_data)
    pca_data_2d_for_plot = re_append_Li_and_ID(pca_data_2d, data)
    save_annotated_plot(pca_data_2d_for_plot, "../figures/pca_2d", '2-component PCA','PC1','PC2', figure_resolution, annotate=True,font_size=16)

    tsne_data_2d = run_2d_tsne(scaled_data_no_Li, seed)
    tsne_data_2d_for_plot = re_append_Li_and_ID(tsne_data_2d, data)
    save_annotated_plot(tsne_data_2d_for_plot, "../figures/tsne_2d", '2-component t-SNE','PC1','PC2', figure_resolution, annotate=True,font_size=16)

    #Modeling and model evaluation

    #Note: you can choose others, or specify a range of hyperparameter
    #values that you want to experiment with.

    models_to_test = {}

    models_to_test['model1'] = LinearRegression()

    regr_cv = RandomForestRegressor(random_state=seed)
    parameters = {
        'n_estimators': [80],
        'criterion': ['absolute_error']
        }
    models_to_test['model2'] = GridSearchCV(regr_cv, parameters, cv=5)

    regr_cv = SVR()
    parameters = {
        'C': [0.2],
        'kernel': ['linear'],
        'epsilon': [1.8]
        }
    models_to_test['model3'] = GridSearchCV(regr_cv, parameters, cv=5)

    regr_cv = KNeighborsRegressor()
    parameters = { 'n_neighbors': [1] }
    models_to_test['model4'] = GridSearchCV(regr_cv, parameters, cv=5)

    model_evaluations = test_all_models(models_to_test, numerical_data, seed)
    save_plots_for_all_models(model_evaluations, "../figures/all_models_mae_full_data", figure_resolution)
    save_plots_for_all_models(model_evaluations, "../figures/all_models_r_squared_full_data", figure_resolution, metric='r_squared')

    #Specific deep dive into one model (in this case, the RF)

    rf_model_evaluations = model_evaluations['model2']
    median_rf = retrieve_median_model_evals(rf_model_evaluations)

    y_test = median_rf['y_test']
    residuals = median_rf['residuals']
    y_pred = median_rf['y_pred']
    y_test = median_rf['y_test']
    feature_names = median_rf['importances_names']
    feature_importances = median_rf['importance_vals']
    X_train = median_rf['X_train']
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model = median_rf['model']
    model_mae = median_rf['mae']
    model_r_squared = median_rf['r_squared']

    save_y_test_histogram(y_test, "../figures/y_test_histogram", figure_resolution)
    save_residuals_histogram(residuals, "../figures/residuals_histogram", figure_resolution)
    save_y_pred_vs_y_test(y_pred, y_test, "../figures/y_pred_vs_y_test", figure_resolution)
    save_residuals_vs_y_test(residuals, y_test, "../figures/residuals_vs_y_test", figure_resolution)
    save_feature_importance_plots(feature_names, feature_importances, "../figures/feature_importance_plot", figure_resolution)
    save_partial_dependence_plot(model, X_train, feature_names, "../figures/partial_dependence_plots", figure_resolution)

    print("Feature selection run complete.")
    print("Median RF MAE: ", str(model_mae))
    print("Median RF r_squared: ", str(model_r_squared))
    print("Median RF importances, all features:")
    for feature_index in np.arange(len(feature_names_rf)):
        print(feature_names_rf[-feature_index], ": ", feature_importances_rf[-feature_index])

    print("***")

    #Feature selection
    num_top_features = 6
    top_features = list(feature_names)[-num_top_features:]
    feautres_to_select = top_features + ['d7Li']
    reduced_data = select_features(numerical_data, feautres_to_select)

    #Repeat the training and evaluation process with the selected features

    model_evaluations = test_all_models(models_to_test, reduced_data, seed)
    save_plots_for_all_models(model_evaluations, "../figures/all_models_mae_top_6", figure_resolution)

    rf_model_evaluations = model_evaluations['model2']
    lr_model_evaluations = model_evaluations['model1']
    svr_model_evaluations = model_evaluations['model3']
    knn_model_evaluations = model_evaluations['model4']

    median_rf = retrieve_median_model_evals(rf_model_evaluations)
    median_lr = retrieve_median_model_evals(lr_model_evaluations)
    median_svr = retrieve_median_model_evals(svr_model_evaluations)
    median_knn = retrieve_median_model_evals(knn_model_evaluations)

    feature_names_lr = median_lr['importances_names']
    feature_importances_lr = median_lr['importance_vals']
    model_rf = median_rf['model']
    model_mae_lr = median_lr['mae']
    model_r_squared_lr = median_lr['r_squared']

    feature_names_svr = median_svr['importances_names']
    feature_importances_svr = median_svr['importance_vals']
    model_svr = median_svr['model']
    model_mae_svr = median_svr['mae']
    model_r_squared_svr = median_svr['r_squared']

    feature_names_knn = median_knn['importances_names']
    feature_importances_knn = median_knn['importance_vals']
    model_knn = median_knn['model']
    model_mae_knn = median_knn['mae']
    model_r_squared_knn = median_knn['r_squared']

    feature_names_rf = median_rf['importances_names']
    feature_importances_rf = median_rf['importance_vals']
    model_rf = median_rf['model']
    model_mae_rf = median_rf['mae']
    model_r_squared_rf = median_rf['r_squared']

    model_mae_mean = median_rf['mae_mean']

    y_test = median_rf['y_test']
    residuals = median_rf['residuals']
    y_pred = median_rf['y_pred']
    y_test = median_rf['y_test']
    X_train = median_rf['X_train']
    X_train = pd.DataFrame(X_train, columns=feature_names_rf)

    save_y_test_histogram(y_test, "../figures/y_test_histogram_top_6", figure_resolution)
    save_residuals_histogram(residuals, "../figures/residuals_histogram_top_6", figure_resolution)
    save_y_pred_vs_y_test(y_pred, y_test, "../figures/y_pred_vs_y_test_top_6", figure_resolution)
    save_residuals_vs_y_test(residuals, y_test, "../figures/residuals_vs_y_test_top_6", figure_resolution)
    save_feature_importance_plots(feature_names_rf, feature_importances_rf, "../figures/feature_importance_plot_top_6", figure_resolution)
    save_partial_dependence_plot(model_rf, X_train, feature_names_rf, "../figures/partial_dependence_plots_top_6", figure_resolution)

    print("Final run complete.")

    print("Median RF MAE: ", str(model_mae_rf))
    print("Median RF r_squared: ", str(model_r_squared_rf))

    for feature_index in np.arange(len(feature_names_rf)):
        print(feature_names_rf[-feature_index], ": ", feature_importances_rf[-feature_index])

    print("***")

    print("Median LR MAE: ", str(model_mae_lr))
    print("Median LR r_squared: ", str(model_r_squared_lr))

    for feature_index in np.arange(len(feature_names_lr)):
        print(feature_names_lr[-feature_index], ": ", feature_importances_lr[-feature_index])

    print("***")

    print("Median SVR MAE: ", str(model_mae_svr))
    print("Median SVR r_squared: ", str(model_r_squared_svr))

    for feature_index in np.arange(len(feature_names_svr)):
        print(feature_names_svr[-feature_index], ": ", feature_importances_svr[-feature_index])

    print("***")

    print("Median KNN MAE: ", str(model_mae_knn))
    print("Median KNN r_squared: ", str(model_r_squared_knn))

    for feature_index in np.arange(len(feature_names_knn)):
        print(feature_names_knn[-feature_index], ": ", feature_importances_knn[-feature_index])

    print("***")

    print("Predict the mean MAE: ", str(model_mae_mean))
