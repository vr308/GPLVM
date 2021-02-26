#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Utilities for loading unsupervised datasets for testing with the GPLVM

"""

import torch
import numpy as np
import sklearn.datasets as skd
from data.data import data_path, resource, dependency

def float_tensor(X): return torch.tensor(X).float()

def raw_movies_data(size):

    import pandas as pd
    
    def _fetch(url, folder):
            resource(target=data_path(folder,  folder + '.zip'),
                     url=url)
            dependency(target=data_path(folder, 'ml'),
                       source=data_path(folder,  folder + '.zip'),
                       commands=['unzip ' + folder + '.zip' + ' -d ' + data_path(folder,'')])
            
    if size == '100k':
        folder = 'movie_lens_100k'
        url =  'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        _fetch(url, folder)

        data = pd.read_csv(data_path(folder) + '/ml-100k/u.data', sep='\t', names=[
                       'userId', 'itemId', 'rating', 'timestamp'])

        movies = pd.read_csv(data_path(folder) + '/ml-100k/u.item', sep='|', names=[
                'itemId', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'],
        encoding='latin')
      
    elif size == '1m':
        folder = 'movie_lens_1m'
        url =  'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        _fetch(url, folder)
        data = pd.read_csv(data_path(folder) + '/ml-1m/ratings.dat', sep='::', names=[
                           'userId', 'itemId', 'rating', 'timestamp'])
    
        movies = pd.read_csv(data_path(folder) + '/ml-1m/movies.dat', sep='::', names=[
                    'itemId', 'title', 'genre'],
            encoding='latin')
        movies = pd.merge(movies, movies['genre'].str.split('|', expand=True), left_index=True, right_index=True)

    return data, movies

def load_unsupervised_data(dataset_name):
    
    '''Loads a range of unsupervised datasets.

    Parameters
    ----------
    dataset_name : str
        One of 'iris' (4D), 'oilflow' (12D), 'gene' (48D), 'mnist' (784D),
        'brendan_faces' (560D), 'movie_lens_100k' (1682D), 'movie_lens_1m' (3706)

    Returns
    -------
    Y : torch.tensor
        Dataset. Return shape is (n x d).
    n : int
        Number of datapoints.
    d : int
        Number of dataset dimensions. 
    labels : numpy.array
        Data categories/classes.

    '''

    if dataset_name == 'iris':
        iris_data = skd.load_iris()
        Y = float_tensor(iris_data.data)
        labels = iris_data.target

    elif dataset_name == 'oilflow':
        
        def _fetch():
            # Training data:
            resource(target=data_path('oilflow', 'oilflow.tar.gz'),
                     url='http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz')
            dependency(target=data_path('oilflow', 'oilflow'),
                       source=data_path('oilflow', 'oilflow.tar.gz'),
                       commands=['tar xzf oilflow.tar.gz'])
        
        _fetch()
        Y = float_tensor(np.loadtxt(data_path('oilflow','') + '/DataTrn.txt'))
        labels = np.loadtxt(data_path('oilflow','') + '/DataTrnLbls.txt')
        labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)

    elif dataset_name == 'gene':
        
        import pandas as pd
        URL = 'https://raw.githubusercontent.com/sods/ods/master/' +\
            'datasets/guo_qpcr.csv'
        gene_data = pd.read_csv(URL, index_col=0)
        Y = float_tensor(gene_data.values)
        raw_labels = np.array(gene_data.index)

        d = dict()
        i = 0
        for label in raw_labels:
            if label not in d:
                d[label] = i
                i += 1
        labels = [d[x] for x in raw_labels]

    elif dataset_name == 'mnist':
        
        from tensorflow.keras.datasets.mnist import load_data
        
        (y_train, train_labels), (y_test, test_labels) = load_data()
        labels = np.hstack([train_labels, test_labels])
        n = len(labels)
        Y = np.vstack([y_train, y_test])
        Y = float_tensor(Y.reshape(n, -1))

    elif dataset_name == 'brendan_faces':

        import pods
        Y = float_tensor(pods.datasets.brendan_faces()['Y'])
        labels = None
    
    elif dataset_name == 'movie_lens_100k':

        # movies = movies.loc[
        #    (movies.Sci_Fi == 1) | (movies.Romance == 1), 'itemId'].tolist()
        # data = data.loc[data.itemId.isin(movies)]

        data, movies = raw_movies_data()

        Y = data.pivot_table(index='userId', columns='itemId',
                             values='rating')

        labels = None
        Y = float_tensor(np.array(Y))
        
    elif dataset_name == 'movie_lens_1m':
        
        data, movies = raw_movies_data('1m')

        Y = data.pivot_table(index='userId', columns='itemId',
                             values='rating')

        labels = None
        Y = float_tensor(np.array(Y))
        
    elif dataset_name == 'movie_lens_100k':

        # movies = movies.loc[
        #    (movies.Sci_Fi == 1) | (movies.Romance == 1), 'itemId'].tolist()
        # data = data.loc[data.itemId.isin(movies)]

        data, movies = raw_movies_data('100k')

        Y = data.pivot_table(index='userId', columns='itemId',
                             values='rating')

        labels = None
        Y = float_tensor(np.array(Y))

    else:
        raise NotImplementedError(str(dataset_name) + ' data not implemented')

    n = len(Y)
    d = len(Y.T)
    return Y, n, d, labels