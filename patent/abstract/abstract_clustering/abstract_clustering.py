import pickle
import json
import hdbscan
from tqdm import tqdm
import ipdb
import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)
import os
TOTAL_NUMBER = int(os.environ.get('TOTAL_NUMBER'))
ABSTRACT_CLUSTER_MIN_NUM = int(os.environ.get('ABSTRACT_CLUSTER_MIN_NUM'))


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def title_centroid(ranked_phrase_file, cpc_title_phrase_embedding_file, output_file):
    phrase_embedding = load_obj(cpc_title_phrase_embedding_file)
    process_data = []
    output_data = {}
    centroids = []
    zero_count = 0
    with open(ranked_phrase_file, 'r', encoding='utf-8') as f_in:
        for item in tqdm(f_in, total=TOTAL_NUMBER):
            item = json.loads(item)
            if len(item) == 0:
                continue
            for key_phrase in item:
                if key_phrase not in phrase_embedding:
                    print('There is an error: cannot find: ' + key_phrase)
                    ipdb.set_trace()
                else:
                    if not (phrase_embedding[key_phrase] == 0.0).all():
                        process_data.append(phrase_embedding[key_phrase])
                    else:
                        print('count: ' + str(zero_count) + 'zero embedding phrase: ' + key_phrase)
                        zero_count = zero_count + 1
                break

    cluster_er = hdbscan.HDBSCAN(min_cluster_size=ABSTRACT_CLUSTER_MIN_NUM)
    cluster_labels = cluster_er.fit_predict(np.array(process_data))
    for item in range(len(cluster_labels)):
        if cluster_labels[item] not in output_data:
            output_data[cluster_labels[item]] = []
        output_data[cluster_labels[item]].append(process_data[item])
    for key in output_data:
        centroids.append(np.mean(output_data[key], axis=0))
    # ipdb.set_trace()
    print('number of centroids: ' + str(len(centroids)))
    save_obj(centroids, output_file)


if __name__ == '__main__':
    ranked_phrase_file = 'patent/abstract/abstract_rank/ranked_abstract_influence_phrase_score_text.json'
    cpc_title_phrase_embedding_file = 'patent/abstract/abstract_embedding/cpc_title_abstract_phrase_embedding.pkl'
    output_file = 'patent/abstract/abstract_clustering/abstract_influence_phrase_centroids.pkl'
    title_centroid(ranked_phrase_file, cpc_title_phrase_embedding_file, output_file)
