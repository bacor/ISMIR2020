"""
Compute 2D embeddings of TF-IDF vectors in all conditions.

Author: Bas Cornelissen
"""
import time
import logging
import os
import sys

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from .helpers import ROOT_DIR, DATA_DIR, RESULTS_DIR
from .helpers import GENRES, SUBSETS
from .helpers import param_str
from .helpers import relpath
from .tfidf_experiment import get_conditions
from .tfidf_experiment import load_dataset
from .tfidf_experiment import get_vectorizer
from .tfidf_experiment import CLASSIFIERS

# Fix random seed
np.random.seed(1234)

EMBEDDERS = ['pca', 'svc', 'tsne']

def cm2inch(*args):
    return list(map(lambda x: x/2.54, args))

def pca_embeddings(X):
    """"""
    t0 = time.time()
    logging.info('Computing PCA embeddings...')

    # Model
    vectorizer = get_vectorizer()
    todense = FunctionTransformer(lambda x: x.todense(), 
        accept_sparse=True, validate=False)
    pca = PCA(n_components=2)
    visualizer = make_pipeline(vectorizer, todense, pca)

    # Fit!
    embeddings = visualizer.fit_transform(X)

    duration = time.time() - t0
    logging.info(f'> Done in {duration:.2f}s')
    return embeddings

def svd_embeddings(X, tfidf_params=dict()):
    """"""
    t0 = time.time()
    logging.info('Computing SVD embeddings...')

    # Model
    vectorizer = get_vectorizer(**tfidf_params)
    svd = TruncatedSVD(n_components=2, random_state=1)
    visualizer = make_pipeline(vectorizer, svd)

    # Fit!
    embeddings = visualizer.fit_transform(X)

    duration = time.time() - t0
    logging.info(f'> Done in {duration:.2f}s')
    return embeddings

def tsne_embeddings(X,
    include_svd=True, n_svd_components=50,
    tsne_params=dict()):
    """
    Computes t-SNE embeddings of the tf-idf vectors. By default it first performs
    a dimensionaly reduction using truncated SVD, as recommended in the scikit-learn
    documentation. This indeed seems to give better results (visually and computationally).
    """
    tsne_default_params = dict(
        n_components=2, 
        n_iter=1000, 
        init='random',
        learning_rate=200,
        perplexity=30,
        random_state=1)
    tsne_default_params.update(tsne_params)
    tsne_params = tsne_default_params
    
    t0 = time.time()
    logging.info('Computing t-SNE embeddings...')
    logging.info(f'> t-SNE: {param_str(tsne_params)}')
    logging.info(f'> include_svd={include_svd}, num_svd_components={n_svd_components}')

    # Build model: only include SVD if the number of features is sufficiently high
    vectorizer = get_vectorizer()
    X_test = vectorizer.fit_transform(X)
    if X_test.shape[1] < n_svd_components:
        include_svd = False
        logging.warning(f'> Excluding SVD: n_features={X_test.shape[1]}')
    else:
        logging.info(f'> n_features={X_test.shape[1]}')

    tsne = TSNE(**tsne_params)
    if include_svd:
        svd = TruncatedSVD(n_components=n_svd_components, random_state=2)
        visualizer = make_pipeline(vectorizer, svd, tsne)
    else:
        todense = FunctionTransformer(lambda x: x.todense(), accept_sparse=True, validate=False)
        visualizer = make_pipeline(vectorizer, todense, tsne)

    # Fit!
    embeddings = visualizer.fit_transform(X)

    duration = time.time() - t0
    logging.info(f'> Done in {duration:.2f}s')
    return embeddings

def compute_embeddings(embedder, genre, subset, representation, segmentation,
    split, data_dir, output_dir, num_datapoints=-1, refresh=False):
    """"""
    logging.info('-'*60)
    logging.info(f'Computing embeddings for segmentation={segmentation} and representation={representation}')
    logging.info(f'> data={relpath(data_dir)}, split={split}, output_dir={relpath(output_dir)}, num_datapoints={num_datapoints}')
    
    # Set up directory/files
    embeddings_dir = os.path.join(output_dir, genre, subset, representation)
    if not os.path.exists(embeddings_dir): 
        os.makedirs(embeddings_dir)

    embeddings_fn = os.path.join(embeddings_dir, f'{embedder}-{split}-{segmentation}.csv')
    if not refresh and os.path.exists(embeddings_fn):
        logging.info(f'> Skipping; file already exists: {relpath(embeddings_fn)}')
        return

    # Load data
    data, modes = load_dataset(
        genre=genre, subset=subset, representation=representation, 
        segmentation=segmentation, data_dir=data_dir, split=split)
    data = data[:num_datapoints]
    modes = modes[:num_datapoints]
        
    # Run!
    try:
        if embedder == 'pca':
            embeddings = pca_embeddings(data)
        elif embedder == 'tsne':
            embeddings = tsne_embeddings(data)
        elif embedder == 'svd':
            embeddings = svd_embeddings(data)
    except Exception as e:
        logging.error(f'> ERROR: {e}')
        return
    
    # Store embeddings as csv file
    df = pd.DataFrame(embeddings, columns=['x', 'y'], index=data.index)
    df['mode'] = modes
    df.to_csv(embeddings_fn)
    logging.info(f'> Stored to {relpath(embeddings_fn)}')

def show_embedding(X, modes, show_legend=False, use_mode_names=False, scale=1, 
    **kwargs):
    colors = sns.color_palette(n_colors=4)
    
    # Labels
    if use_mode_names:
        labels = ['dorian', 'hypodorian', 'phrygian', 'hypophrygian', 
                'lydian', 'hypolydian', 'mixolydian', 'hypomixolydian']
        ncol = 1
        mode_order = range(1, 9)
    else:
        labels = [str(i) for i in range(1,9)]
        ncol = 2
        mode_order = [1, 3, 5, 7, 2, 4, 6, 8]

    for i in mode_order:
        is_mode = modes == i
        xs = X[is_mode, 0]
        ys = X[is_mode, 1]     
        
        # Plot properties
        props = dict(alpha=.5)
        props.update(kwargs)
        props['marker'] = 'o' if i % 2 == 0 else 'x'
        props['mew']    = 0   if i % 2 == 0 else .5
        props['ms']     = scale * 1.5 if i % 2 == 0 else scale * 2
        props['color']  = colors[int(np.ceil(i / 2)) - 1]
        props['label']  = labels[i-1]
        
        # Plot!
        plt.plot(xs, ys, lw=0, **props)

    plt.axis('off')
    plt.gca().get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().set_aspect('equal')
    if show_legend:
        leg = plt.legend(bbox_to_anchor=(1.0,1), loc="upper left", frameon=False, ncol=ncol, fontsize=8)
        for lh in leg.legendHandles: 
            lh._legmarker.set_alpha(1)

def visualize_embedding(embedder, genre, subset, representation, segmentation,
    split, embeddings_dir, output_dir, refresh=False):
    logging.info('-'*60)
    logging.info(f'Visualizing embeddings for segmentation={segmentation} and representation={representation}')
    logging.info(f'> embeddings_dir={relpath(embeddings_dir)}, output_dir={relpath(output_dir)}')
    
    embeddings_fn = os.path.join(
        embeddings_dir, genre, subset, representation, f'{embedder}-{split}-{segmentation}.csv')
    plot_dir = os.path.join(output_dir, genre, subset, representation)
    plot_fn = os.path.join(plot_dir, f'{embedder}-{split}-{segmentation}.png')
    if not os.path.exists(plot_dir): 
            os.makedirs(plot_dir)
    
    if not refresh and os.path.exists(plot_fn):
        logging.info(f'> Skipping; file already exists: {plot_fn}')
    elif not os.path.exists(embeddings_fn):
        logging.warning(f'Embedding file {relpath(embeddings_fn)} not found')
    else:
        # Load embeddings
        df = pd.read_csv(embeddings_fn)

        # Plotting
        logging.info('Plotting representations...')
        plt.figure(figsize=cm2inch(5,5))
        show_embedding(df[['x', 'y']].values, df['mode'], show_legend=False)
        plt.tight_layout(pad=0, w_pad=0)
        plt.savefig(plot_fn, dpi=400, transparent=True)
        plt.close()
        logging.info(f'> Stored figure to {relpath(plot_fn)}') 

def run(experiment_name,
        embedders='all', genres='all', subsets='all', 
        representations='all', segmentations='all',
        splits='all', refresh=False,
        data_dir = DATA_DIR, results_dir = RESULTS_DIR):
    
    # Defaults
    embedders = EMBEDDERS if embedders == 'all' else embedders
    splits = ['train', 'test'] if splits == 'all' else splits
    results_dir = os.path.join(results_dir, f'embeddings-{experiment_name}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set up logging
    log_fn = os.path.join(results_dir, f'embeddings-experiment-{experiment_name}.log')
    print(f'Starting log: {relpath(log_fn)}')
    logging.basicConfig(
        filename=log_fn,
        filemode='w',
        format='%(levelname)s %(asctime)s %(message)s',
        datefmt='%d-%m-%y %H:%M:%S',
        level=logging.INFO) 

    conditions, parts = get_conditions(
        CLASSIFIERS[0], genres, subsets, representations, segmentations)
    for condition in conditions:
        del condition['classifier']
        for embedder in embedders:
            for split in splits:
                compute_embeddings(
                    embedder=embedder, 
                    split=split,
                    refresh=refresh,
                    output_dir=os.path.join(results_dir, 'embeddings'),
                    data_dir=DATA_DIR,
                    **condition)
                visualize_embedding(
                    embedder=embedder, 
                    split=split,
                    refresh=refresh,
                    embeddings_dir=os.path.join(results_dir, 'embeddings'),
                    output_dir=os.path.join(results_dir, 'figures'),
                    **condition)

if __name__ == '__main__':
    run('tfidf-tune-20-iter',
        embedders=['pca', 'tsne'],
        genres=['responsory'],
        subsets=['full'],
        splits=['test'],
        refresh=True,
        # representations=['contour-independent'],
        # segmentations=['words', 'syllables', '6-mer']
        )