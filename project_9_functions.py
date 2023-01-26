# functions used in oc_project_9

import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity



def get_clicks_df(test_mode, clicks_sample_path, clicks_dir_path):
    clicks_sample = pd.read_csv(clicks_sample_path)
    
    if test_mode:
        return clicks_sample

    clicks_list = []
    column_names = list(clicks_sample.columns)
    clicks = pd.DataFrame(columns=column_names)
    n = len(os.listdir(clicks_dir_path))
    pad_len = len(str(n))
    
    for i in range(n):
        clicks_i_path = clicks_dir_path + 'clicks_hour_' + str(i).zfill(pad_len) + '.csv'
        clicks_i = pd.read_csv(clicks_i_path)
        clicks_list.append(clicks_i)

    clicks = pd.concat(clicks_list, axis=0)
    return clicks



def get_embeddings(embeddings_path, test_mode, pca=0):
    embeddings = np.load(embeddings_path, allow_pickle=True)

    if test_mode:
        n_test_samples = 1000
        embeddings = embeddings[:n_test_samples,:]

    if pca:
        pca = PCA(n_components=pca, random_state=0)
        embeddings = pca.fit_transform(embeddings)
    
    return embeddings



def extract_with_indices(x, idx):
    return x[np.arange(x.shape[0])[:, None], idx]



def get_recommended_indices(n, embeddings, method, params, sorted=True):
    if method == 'last_seen':
        i = params['last_seen']

        # compute similarities
        embedding_rows = embeddings[i,:].reshape(1, -1)
        embeddings_without_i = np.delete(embeddings, i, axis=0)
        res = cosine_similarity(embedding_rows, embeddings_without_i)
    
        # get top "n" similarities
        top_n_indices = np.argpartition(res,-n)[:,-n:]
        if sorted:
            top_n_values = extract_with_indices(res, top_n_indices)
            top_n_indices = extract_with_indices(top_n_indices, np.flip(np.argsort(top_n_values), axis=1))
            #top_n_values = extract_with_indices(res, top_n_indices)

        return top_n_indices

    else:
        return '"method" is not valid.'



def get_all_recommendations(n, embeddings, method, params, chunk_size, sorted=False, verbose=False):
    if method == 'last_seen':
        nb_articles = embeddings.shape[0]
        error_message = 'chunk_size too big, must be <= nb_articles - n'
        assert chunk_size <= nb_articles - n, error_message
        
        for i in range(0, nb_articles, chunk_size):
            # set chunk boundaries
            try:
                j = i + chunk_size - 1
                _ = embeddings[j,:]
            except IndexError:
                j = nb_articles - 1
            
            if verbose:
                if i == chunk_size:
                    print('first iteration finished.')
                
        
            # compute chunk similarities
            embedding_rows = embeddings[i:j+1,:]
            chunk_indices = np.arange(i, j+1)
            embeddings_without_chunk = np.delete(embeddings, chunk_indices, axis=0)
            res = cosine_similarity(embedding_rows, embeddings_without_chunk)
        
            # get top "n" similarities
            top_n_indices = np.argpartition(res,-n)[:,-n:]
            if sorted:
                top_n_values = extract_with_indices(res, top_n_indices)
                top_n_indices = extract_with_indices(top_n_indices, np.flip(np.argsort(top_n_values), axis=1))
                #top_n_values = extract_with_indices(res, top_n_indices)                
            
            # stack chunk results in one array
            try:
                recs = np.vstack([recs, top_n_indices])
            except UnboundLocalError:
                recs = top_n_indices
        return recs
    
    else:
        return '"method" is not valid.'



def get_user_read_list(user_id, time_click_df):
    group_user = time_click_df[time_click_df['user_id'] == user_id]
    sorted_group = group_user.sort_values('click_timestamp', ascending=False)
    read_list = list(sorted_group['click_article_id'])
    return read_list
