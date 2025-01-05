import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .plot_settings import format_settings


def plot_single_show_number(data):
    fig = plt.figure(figsize=(12, 8))
    format_settings(ms=5, major_tick_len=0, fs=8, axlw=0)
    
    im = plt.imshow(data, cmap='viridis')
    cbar = plt.colorbar(im)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='w')


def plot_single(data, title='123', savename='123', close = True):
    if data.ndim == 2:
        fig = plt.figure(figsize=(12, 8))
        format_settings(ms=5, major_tick_len=0, fs=8, axlw=0)
        ax = plt.gca()
        ax.set_title(title)
        im = ax.imshow(data, cmap='viridis')
        cbar = plt.colorbar(im, orientation='horizontal')
        plt.savefig(f'zzz_3x_to_x_2layer1head_analysis/{savename}.png', dpi=300)
        if close:
            plt.close()


def pca(X0):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X0)
    return X

def tsne(X0):
    tsne = TSNE(n_components=2)
    X = tsne.fit_transform(X0)
    return X

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cosine_similarity_array(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return np.dot(X, X.T)


def graph_entropy(A):
    A = A / np.sum(A, axis=1, keepdims=True)
    A = A + 1e-8
    return -np.sum(A * np.log(A), axis=1)