import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def generate_correlation_matrix(data):

    #We square the correlation matrix values, so that negative and positive
    #correlations look the same in the matrix.
    corr_matrix = data.corr()**2
    return corr_matrix

def save_correlation_matrix(matrix, path):

    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = 'Greens'

    sns.heatmap(matrix,
        cmap=cmap,
        square=True,
        linewidth=.5,
        cbar_kws={"shrink": .5},
        ax=ax)

    plt.savefig(path, dpi=150)
    plt.clf()

def save_correlations_with_d7Li(correlation_matrix, path):

    plt.figure(figsize=(12,4))
    correlation_matrix['d7Li'].plot.bar(color='Blue')
    plt.title('Squared Pearson correlations with d7Li')

    plt.savefig(path, dpi=150)
    plt.clf()

def save_scatter_matrix(data, path):
    pd.plotting.scatter_matrix(data, alpha=0.8, figsize=(20,20))
    plt.savefig(path, dpi=150)
    plt.clf()

def save_partial_scatter_matrix(data, columns, path):
    colors = []
    colors = ['blue' for sample in np.arange(len(data))]
    pd.plotting.scatter_matrix(data[columns], alpha=0.8, figsize=(10,10), color=colors, s=500)
    plt.savefig(path, dpi=150)
    plt.clf()

def remove_Li(data):
    data_no_Li = data.drop(['d7Li'], axis=1)
    return data_no_Li

def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

def run_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)

    explained_variance_ratios = pca.explained_variance_ratio_

    return pca_data, explained_variance_ratios, pca

def save_pca_component_contributions(pca, data, component, path):
    feature_contributions = np.abs(pca.components_[component])
    sorted_indices = np.argsort(feature_contributions)
    feature_names = list(data.columns[sorted_indices])
    contributions = feature_contributions[sorted_indices]

    contribution_list = [{'feature': feature_names[i], 'contribution': contributions[i]} for i in sorted_indices]
    sorted(contribution_list, key=lambda k: k['contribution'])

    fig, ax = plt.subplots(figsize=(12,4))

    ax.bar(feature_names, contributions)
    plt.xticks(rotation=90)
    plt.title('Principal component ' + str(component + 1))
    plt.savefig(path + str(component + 1), dpi=150)

    plt.gcf().subplots_adjust(bottom=0.8)

    plt.clf()

def save_all_pca_component_contributions(pca, data, n_components, path):
    for component in np.arange(n_components):
        save_pca_component_contributions(pca, data, component, path)

def save_explained_variance_ratios(explained_variance_ratios, path):
    fig, ax = plt.subplots(figsize=(12,4))

    labels = np.arange(len(explained_variance_ratios))

    ax.bar(labels, explained_variance_ratios)
    plt.savefig(path, dpi=150)
    plt.clf()

def save_cumulative_variance(explained_variance_ratios, path):
    cumulative_variance = np.cumsum(explained_variance_ratios)
    labels = np.arange(len(explained_variance_ratios))

    fig, ax = plt.subplots(figsize=(12,4))

    ax.bar(labels, explained_variance_ratios)
    plt.plot(cumulative_variance)
    plt.savefig(path, dpi=150)
    plt.clf()

def generate_2_component_pca_df(data):
    pca_data_2d = data[:, [0, 1]]
    return pd.DataFrame(pca_data_2d, columns=['PC1', 'PC2'])

def re_append_Li_and_ID(data_no_Li_no_ID, data):
    return pd.concat([data_no_Li_no_ID, data['d7Li'], data['Sample_ID']], axis=1)

def save_annotated_plot(data, path,
    title, x_label, y_label,
    annotate=True,
    font_size=16):

    scaler = MinMaxScaler()
    colors = scaler.fit_transform(data[['d7Li']])
    norm = mpl.colors.Normalize()
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_map = m.to_rgba(np.reshape(colors, -1))

    fig = plt.figure(figsize = (12,12))

    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_title(title, fontsize=font_size, y=1.04)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    x_components = data[x_label]
    y_components = data[y_label]

    marker_map = ['x' for i in np.arange(data.shape[0])]

    for _s, c, _x, _y in zip(marker_map, color_map, x_components, y_components):
        plt.scatter(_x, _y, marker=_s, c=c)

    if annotate:
        for i, sample_id in enumerate(list(data['Sample_ID'])):
            ax.annotate(str(' ' + sample_id), (x_components[i], y_components[i]),
            color=color_map[i], rotation=0, fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    ax.grid()
    plt.savefig(path, dpi=150)
    plt.clf()

def run_2d_tsne(data):

    #Note: the perplexity value below is a parameter that needs to be
    #modified for each new dataset.
    tsne = TSNE(n_components=2, perplexity=5)
    tsne_data_2d = tsne.fit_transform(data)
    return pd.DataFrame(tsne_data_2d, columns=['PC1', 'PC2'])
