# Applying Machine Learning to Understand Yukon Basin Weathering Processes

## How to run the code

`cd` into the `scripts` folder and simply run the command

```
python3 -i main.py
```

You may see some deprecation warnings depending on your environment,
but you should be able to continue without issue. 

## Description of the data

In the top-level folder, you'll find the dataset used for this project (`data.csv`).
The dataset has been cleaned, such that it contains no empty entries. The first
row of the csv contains the names of all the features as they will appear in the
visualizations and analysis plots that will be generated by the code. There are
21 such features in the dataset as shown here, but you can safely substitute
it with your own, even if it has more or fewer features. If you do, keep two
things in mind:

- The feature titled `d7Li` will be used as the target variable in the modeling
stage of this project.
- The feature titled `Sample_ID` is a unique identifier that represents each
sample. This column is used to tag individual samples in several plots. If
you don't plan on assigning a Sample_ID to each of your samples you may leave
this column blank (but keep the `Sample_ID` header). Note that if you do so,
you will also have to ensure that the `annotate` argument to the `save_annotated_plot`
function is set to `False` (`annotate=True` means that each sample in the plot
  will be tagged with its corresponding `Sample_ID`).

## Description of the codebase

In its top-level folder, the codebase contains the dataset, as well as two folders:
`scripts`, which contains the code that will generate analysis and visualization
plots, and `figures`, which contains the plots themselves.

The `scripts` folder contains the following libraries:

- `data_preprocessing.py`, which contains a small number of data cleaning functions.

- `data_visualization.py`, which contains functions that generate a wide range of
data visualization plots. These include plots of dimensionality-reduced versions
of the data (generated using PCA or t-SNE), as well as a variety of histograms,
pairwise feature comparison scatter plots, and correlation matrices.

- `modeling_and_evaluation.py`, which contains functions that handle model training,
evaluation, and visualization of model performance characteristics. You can read more
about this module in the section below.

In addition to these library files, `scripts` also contains `main.py`, the main
module, which calls on the functions provided in the library files to carry out
the analysis, model-building, and evaluation.

The `figures` folder contains the images generated by the modules in the `scripts` folder.

## Description of `main.py`

### Data cleaning and initial visualization

`main.py` begins by importing the data from `data.csv`, performing a simple
cleaning step, and then generating visualization plots. Among these plots is a
giant scatter matrix (a matrix that displays pairwise scatter plots of each numerical
feature in the dataset). While comprehensive, its size can make detailed study
of specific feature pairs difficult. As a result, a function called `save_partial_scatter_matrix`
is called, which plots only a subset of the full scatter matrix (the specific
features that are displayed in this partial scatter matrix are those listed in the
`columns` argument of `save_partial_scatter_matrix`).

### Dimensionality reduction analysis

Next comes a dimensionality reduction analysis. `main.py` performs 7-component
PCA on the input data (the number of components can be changed by updating the
value of `n_components`). It then generates plots showing explained variance
ratios and feature compositions for every principal component.

Then, 2-component PCA and t-SNE plots are generated, and each sample is colour-coded
according to its `d7Li` value.

Note that the data that was used to perform the dimensionality reduction was
stripped of its `d7Li` feature. This is because we were most interested in using the 2-D
PCA and t-SNE plots to see whether samples with similar `d7Li` values would naturally
cluster together when characterized by their independent variable values alone.
As such, including the `d7Li` values would lead to an artificial clustering effect,
which amounts to a form of data leakage that would exaggerate the apparent
predictive power of 2-D t-SNE or PCA-based clustering.

### Modeling and model evaluation

The modeling process begins with the specification of a set of 4 different
models that will be trained and evaluated on the dataset. They are added one
after the other to the `models_to_test` variable, and include a linear regression
(`model1`), a random forest (`model2`), a support vector machine (`model3`),
and a KNN (`model4`). In each case, models are accompanies by a grid of
hyperparameter values that will be scanned using `GridSearchCV` to identify the
best-performing set of hyperparameters for that model type.

Note that in the current implementation, only a very limited set of hyperparameter
values are being scanned. This is to make the code run faster, greater
hyperparameter volume leads to more computation time, and an initial assessment
of different hyperparameter values had already been performed on each model to
determine roughly which values would be optimal. When applying this script to a new
dataset, we recommend testing a wider range of hyperparameter values for each
model you're interested in testing.

Once the models are defined, they are fed to  the `test_all_models` function,
which generates 30 different train/test splits of the original dataset for each
model of interest. This results in 30 trained and evaluated models, some of which
will be found upon evaluation to perform better than others, purely because
of a favourable train/test split (for example, one that left samples whose `d7Li`
values are particularly easy to predict disproportionately in the testing set).
The distribution of MAEs obtained for these 30 models is then plotted for each of
the 4 model types of interest.

In addition, a "baseline" model is defined, to which all other models can be compared
in order to gauge their predictive power. This baseline model consists of using
the mean `d7Li` value of the samples in the training set as the predictor for the
`d7Li` values in the test set. Just as was done for the 4 models of interest,
this "baseline" model is evaluated across 30 different train/test splits, and
the distribution of its MAE scores is plotted against those of the other 4 models
in `all_models_mae.png`.

### Performance analysis for median random forest model

Because the variation in model performance across the 30 train/test splits described
above is due purely to the chance distribution of samples between the training and
testing sets, we treat the median-performing model as canonical going forward.

We also focus exclusively on the median-performing random forest model, since,
according to the results displayed in `all_models_mae.png`, the RF performs
considerably better than other model types tested in this study.

Model evaluation, feature importance, and partial dependence plots are therefore generated
for the median-performing RF model.

Finally, the MAE and R^2 values obtained for the median RF model are printed
explicitly.
