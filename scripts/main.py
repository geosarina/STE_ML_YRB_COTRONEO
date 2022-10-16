from data_preprocessing import *
from data_visualization import *
from modeling_and_evaluation import *
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':

    #Data preprocessing

    data = read_data('../data.csv')
    cols_to_drop = ['Sample_ID']
    data, numerical_data = drop_columns(data, cols_to_drop)

    #Data visualization

    corr = generate_correlation_matrix(numerical_data)

    save_correlation_matrix(corr, "../figures/correlation_matrix")
    save_correlations_with_d7Li(corr, "../figures/d7Li_correlations")
    save_scatter_matrix(numerical_data, "../figures/scatter_matrix")

    #Showing an example of a pair of columns you might want to compare
    columns_to_compare = ["d7Li", "MAT"]
    save_partial_scatter_matrix(numerical_data, columns_to_compare, "../figures/partial_scatter_matrix")

    #Dimensionality reduction (PCA and t-SNE)

    data_no_Li = remove_Li(numerical_data)
    scaled_data_no_Li = scale_data(data_no_Li)

    n_components = 7
    pca_data, explained_variance_ratios, pca = run_pca(scaled_data_no_Li, n_components)

    save_all_pca_component_contributions(pca, data_no_Li, n_components, "../figures/PC")
    save_explained_variance_ratios(explained_variance_ratios, "../figures/explained_variance_ratios")
    save_cumulative_variance(explained_variance_ratios, "../figures/cumulative_variance")

    pca_data_2d = generate_2_component_pca_df(pca_data)
    pca_data_2d_for_plot = re_append_Li_and_ID(pca_data_2d, data)
    save_annotated_plot(pca_data_2d_for_plot, "../figures/pca_2d", '2-component PCA','PC1','PC2', annotate=True,font_size=16)

    tsne_data_2d = run_2d_tsne(scaled_data_no_Li)
    tsne_data_2d_for_plot = re_append_Li_and_ID(tsne_data_2d, data)
    save_annotated_plot(tsne_data_2d_for_plot, "../figures/tsne_2d", '2-component t-SNE','PC1','PC2', annotate=True,font_size=16)

    #Modeling and model evaluation

    #Note: you can choose others, or specify a range of hyperparameter
    #values that you want to experiment with.

    models_to_test = {}

    models_to_test['model1'] = LinearRegression()

    regr_cv = RandomForestRegressor()
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

    model_evaluations = test_all_models(models_to_test, numerical_data)
    save_plots_for_all_models(model_evaluations, "../figures/all_models_mae")

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
    model_r_square = median_rf['r_square']

    save_y_test_histogram(y_test, "../figures/y_test_histogram")
    save_residuals_histogram(residuals, "../figures/residuals_histogram")
    save_y_pred_vs_y_test(y_pred, y_test, "../figures/y_pred_vs_y_test")
    save_residuals_vs_y_test(residuals, y_test, "../figures/residuals_vs_y_test")
    save_feature_importance_plots(feature_names, feature_importances, "../figures/feature_importance_plot")
    save_partial_dependence_plot(model, X_train, feature_names, "../figures/partial_dependence_plots")

    print("Run complete.")
    print("Median RF MAE: ", str(model_mae))
    print("Median RF r_squared: ", str(model_r_square))
