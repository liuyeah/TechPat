import pickle
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn import linear_model
import math
import random
TOTAL_NUMBER = int(os.environ.get('TOTAL_NUMBER'))


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def score_normalize(original_score_file, output_score_list_file):
    original_score = load_obj(original_score_file)
    normalized_score = []
    for supergraph_dic in tqdm(original_score, total=TOTAL_NUMBER):
        temp_normalized_score = {}
        temp_sum_score = []

        if len(supergraph_dic) == 0:
            normalized_score.append(temp_normalized_score)
            continue
        for subgraph_idx in supergraph_dic:
            subgraph = supergraph_dic[subgraph_idx]
            temp_normalized_score[subgraph_idx] = {}
            for node in subgraph:
                temp_normalized_score[subgraph_idx][node] = np.sum(subgraph[node])
                temp_sum_score.append(np.sum(subgraph[node]))
                # print('1')
        temp_max = np.max(temp_sum_score)
        temp_min = np.min(temp_sum_score)
        score_bottom = temp_max - temp_min
        for subgraph_idx in temp_normalized_score:
            subgraph = temp_normalized_score[subgraph_idx]
            for node in subgraph:
                score_top = subgraph[node] - temp_min
                if score_bottom == 0:
                    subgraph[node] = 1.0
                else:
                    subgraph[node] = score_top/score_bottom
        normalized_score.append(temp_normalized_score)
    save_obj(normalized_score, output_score_list_file)



if __name__ == '__main__':
    original_score_file = 'patent/abstract/abstract_score/abstract_influence_phrase_score.pkl'
    output_score_list_file = 'patent/abstract/abstract_score/abstract_influence_phrase_list_normalized_score.pkl'
    score_normalize(original_score_file, output_score_list_file)
