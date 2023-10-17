import hdbscan
import pickle
import ipdb
import numpy as np
from sklearn.datasets import make_blobs
import logging
logging.basicConfig(level=logging.DEBUG)
import os
EMBEDDING_SIZE = int(os.environ.get('EMBEDDING_SIZE'))
CPC_CLUSTER_MIN_NUM = int(os.environ.get('CPC_CLUSTER_MIN_NUM'))


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def search_phrase_embedding(phrase, embedding_table):
    # item = phrase.strip()
    temp_item = phrase.lower()
    if temp_item in embedding_table:
        return embedding_table[temp_item]
    else:
        print('There is an error: cannot find: ' + temp_item)
        # ipdb.set_trace()
    return np.zeros(EMBEDDING_SIZE, dtype='float32')


def calculate_centroid(groudtruth_file, phrase_embedding_file, centroids_file):
    processed_data = []
    output_data = {}
    centroids = []
    groudtruth = []
    with open(groudtruth_file, 'r', encoding='utf-8') as f_in:
        for i in f_in:
            groudtruth.append(i.strip('\n'))
    embedding_table = load_obj(phrase_embedding_file)
    for i in groudtruth:
        if i != '':
            if not (search_phrase_embedding(i, embedding_table) == 0.0).all():
                processed_data.append(search_phrase_embedding(i, embedding_table))
            else:
                print('zero embedding phrase: ' + i)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=CPC_CLUSTER_MIN_NUM)
    cluster_labels = clusterer.fit_predict(np.array(processed_data))
    for item in range(len(cluster_labels)):
        if cluster_labels[item] not in output_data:
            output_data[cluster_labels[item]] = []
        output_data[cluster_labels[item]].append(processed_data[item])
    for key in output_data:
        centroids.append(np.mean(output_data[key], axis=0))
    # ipdb.set_trace()
    print('centroid number: ' + str(len(centroids)))
    save_obj(centroids, centroids_file)




if __name__ == '__main__':
    groudtruth_file = 'example_data/example_cpc/cpc_phrase.txt'
    phrase_embedding_file = 'patent/cpc/cpc_embedding/cpc_phrase_embedding.pkl'
    centroids_file = 'patent/cpc/cpc_clustering/cpc_centroids.pkl'
    calculate_centroid(groudtruth_file, phrase_embedding_file, centroids_file)

