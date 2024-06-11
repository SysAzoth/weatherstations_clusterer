from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.base import clone

from copy import deepcopy

from collections import OrderedDict

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
jax.default_device(jax.devices('cpu')[0])

import jax.numpy as jnp
import numpy as np
import pandas as pd

import itertools

import matplotlib.pyplot as plt

from iris_mog.dag_kl import dag_kl
from iris_mog.better_graphs import process_graph



def visualize(data, y, y_hat):
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # Double the width to accommodate two plots
    ax[0].scatter(data_reduced[:, 0], data_reduced[:, 1], c=y_hat, edgecolor='k', s=50, cmap='rainbow')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    ax[0].set_title('PCA, Predicted Labels by GMM')
    ax[0].grid(True)
    scatter = ax[1].scatter(data_reduced[:, 0], data_reduced[:, 1], c=y, edgecolor='k', s=50, cmap='viridis')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].set_title('PCA Ground Truth')
    ax[1].grid(True)
    plt.show()


def check_epsilons(gmm, n_samples, axes_to_keep):
    data, labels = gmm.sample(n_samples)
    data_castrated = data[:, axes_to_keep]

    gmm_castrated = GaussianMixture(n_components=len(axes_to_keep), covariance_type='diag')

    gmm_castrated.weights_ = gmm.weights_
    gmm_castrated.covariances_ = gmm.covariances_[:, axes_to_keep]
    gmm_castrated.means_ = gmm.means_[:, axes_to_keep]
    gmm_castrated.precisions_cholesky_ = jnp.sqrt(1 / gmm_castrated.covariances_)
    gmm_castrated.converged = True

    p_L_X = gmm.predict_proba(data)
    p_L_Xs = gmm_castrated.predict_proba(data_castrated)

    p_X = np.exp(gmm.score_samples(data))

    #edkl = p_X @ np.einsum("xl,xl->x", p_L_X, (np.log(p_L_X) - np.log(p_L_Xs)))
    edkl = np.einsum("xl,xl->x", p_L_X, (np.log(p_L_X) - np.log(p_L_Xs))).mean()

    edkl = edkl / jnp.log(2)

    if edkl > 50:
        print("woe unto all")

    print("E_x[Dkl(P[(L|X) || (L|Xs)])] = ", edkl)

def generate_combinations(nums):
    for i in range(1, len(nums) + 1):
        for combination in itertools.combinations(nums, i):
            yield combination

def main():
    np_rng = 0
    jax_rng = jax.random.PRNGKey(np_rng)

    n_gmm_components = 3
    covariance_type = 'diag'
    init_params = 'random' # default: 'kmeans'

    data_df = pd.read_csv("iris.csv")
    data = data_df[[c for c in data_df.columns[:-1]]].to_numpy(dtype=np.float32)
    y = data_df['species'].map({"setosa": 0, "versicolor": 1, "virginica": 2}).values

    gmm = GaussianMixture(n_components=n_gmm_components, random_state=np_rng,
                          covariance_type=covariance_type, init_params=init_params).fit(data)
    y_hat = gmm.predict(data)


    for axes_to_keep in generate_combinations(list(range(data.shape[1]))):
        print(axes_to_keep)
        check_epsilons(gmm, n_samples=len(data), axes_to_keep=axes_to_keep)
    visualize(data, y, y_hat)


if __name__ == "__main__":
    main()