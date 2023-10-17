import numpy as np
from tqdm import tqdm
import pickle
import ipdb
import json
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import pdist, squareform, cosine
import os
TOTAL_NUMBER = int(os.environ.get('TOTAL_NUMBER'))
EMBEDDING_SIZE = int(os.environ.get('EMBEDDING_SIZE'))

BASIC_THRESHOLD = 0.5
ALPHA = 0.5


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def most_similar(x, phrase_embedding_idx,  phrase_embedding_distance_matrix, topN=None):
    if x not in phrase_embedding_idx:
        print('cannot find word: ' + str(x))
        ipdb.set_trace()
        if topN:
            return [1]*topN
        else:
            return []
    if len(phrase_embedding_idx) == 1:
        if topN:
            return [1]*topN
        else:
            return []
    idx = phrase_embedding_idx[x]
    output_list = []

    distance_list = sorted(phrase_embedding_distance_matrix[idx], reverse=False)
    if topN:
        if topN >= len(distance_list):
            print('There is an error in most similar function!!!')
            ipdb.set_trace()
        for i in range(1, topN+1):
            output_list.append(distance_list[i])
    else:
        for i in range(1, len(distance_list)):
            output_list.append(distance_list[i])

    return output_list


def scalability(node, phrase_embedding_idx, phrase_embedding_distance_matrix):
    if len(phrase_embedding_idx) == 1:
        return 0
    neighbor_word2sim = [sim for sim in
                         most_similar(node, phrase_embedding_idx,  phrase_embedding_distance_matrix)
                         if sim <= BASIC_THRESHOLD]
    return float(len(neighbor_word2sim))/float(len(phrase_embedding_idx)-1)


def independence(node, phrase_embedding_idx, phrase_embedding_distance_matrix):
    if len(phrase_embedding_idx) == 1:
        return 1
    neighbor_word2sim = most_similar(node, phrase_embedding_idx,  phrase_embedding_distance_matrix, topN=1)
    return neighbor_word2sim[0]


def super_most_similar(x, node_idx, phrase_embedding_distance_matrix, topN=None):
    output_list = []

    distance_list = sorted(phrase_embedding_distance_matrix[node_idx], reverse=False)
    if topN:
        if topN >= len(distance_list):
            print('There is an error in most similar function!!!')
            ipdb.set_trace()
        for i in range(1, topN+1):
            output_list.append(distance_list[i])
    else:
        for i in range(1, len(distance_list)):
            output_list.append(distance_list[i])
    return output_list

def supergraph_independence(node, node_idx, node_embedding, subgraph_idx, subgraph_avg_dic, subsubgraph_distance_matrix):
    if len(subsubgraph_distance_matrix) == 1:
        local_independence = 1
    else:
        neighbor_word2sim = super_most_similar(node, node_idx, subsubgraph_distance_matrix, topN=1)
        local_independence = neighbor_word2sim[0]

    global_independence = 1
    if len(subgraph_avg_dic) == 1:
        global_independence = 1
    else:
        for avg_item in subgraph_avg_dic:
            if avg_item == subgraph_idx:
                continue
            else:
                temp = cosine(node_embedding, subgraph_avg_dic[avg_item])
                if temp < global_independence:
                    global_independence = temp
    return ALPHA*local_independence + (1-ALPHA)*global_independence


def supergraph_relation(node, node_idx, node_embedding, subgraph_idx, subgraph_avg_dic, subsubgraph_distance_matrix):
    if len(subsubgraph_distance_matrix) == 1:
        local_relation = 0
    else:
        neighbor_word2sim = [sim for sim in
                         super_most_similar(node, node_idx, subsubgraph_distance_matrix)
                         if sim <= BASIC_THRESHOLD]
        local_relation = float(len(neighbor_word2sim))/float(len(subsubgraph_distance_matrix)-1)
    if len(subgraph_avg_dic) == 1:
        global_relation = 0
    else:
        global_realtion_count = 0
        for avg_item in subgraph_avg_dic:
            if avg_item == subgraph_idx:
                continue
            else:
                temp = cosine(node_embedding, subgraph_avg_dic[avg_item])
                if temp < BASIC_THRESHOLD:
                    global_realtion_count = global_realtion_count + 1
        global_relation = float(global_realtion_count) / float(len(subgraph_avg_dic)-1)
    return ALPHA*local_relation + (1-ALPHA)*global_relation


def self_score(node):
    if len(node.split(' ')) == 2 or len(node.split(' ')) == 3 or len(node.split(' ')) == 4:
        return 1.0
    elif len(node.split(' ')) == 5:
        return 0.5
    else:
        return 0.0


def occurance(node, sentence_text):
    count = 0
    document_list = sentence_text.split(' ')
    temp_node = node.split(' ')
    if len(document_list) < len(temp_node):
        return 0
    elif len(document_list) == len(temp_node) and document_list[0] != temp_node[0]:
        return 0

    for i in range(len(document_list)-len(temp_node)+1):
        signal = 0
        for j in range(len(temp_node)):
            if document_list[i+j] != temp_node[j]:
                signal = 1
                break
        if signal == 0:
            count = count + 1

    return count


def influence(node, sentence_text):
    influence_count = 0
    sentence_list = sent_tokenize(sentence_text)
    for item in sentence_list:
        if occurance(node, item) == 0:
            continue
        else:
            influence_count = influence_count + 1

    return influence_count



def centroid_score(node_embedding, centroid_list):
    similarity_list = []
    zero_list = np.zeros(EMBEDDING_SIZE, dtype='float32')
    for centroid in centroid_list:
        if (centroid == node_embedding).all():
            similarity_list.append(1.0)
        elif (centroid == zero_list).all() or (node_embedding == zero_list).all():
            similarity_list.append(0.0)
        else:
            similarity_list.append(1 - cosine(node_embedding, centroid))
    return np.max(similarity_list)


def local_topic_score(node, last_level_phrase):
    if node in last_level_phrase:
        return 1
    else:
        return 0


def supergraph_node_score(node, node_idx, node_embedding, subgraph_idx, subgraph_avg_dic, subsubgraph_distance_matrix, 
                            lower_sentence_text, last_level_phrase, centroid_list):
    final_score = [self_score(node),
                   influence(node, lower_sentence_text),
                   local_topic_score(node, last_level_phrase),
                   centroid_score(node_embedding, centroid_list),
                   supergraph_independence(node, node_idx, node_embedding, subgraph_idx, subgraph_avg_dic, subsubgraph_distance_matrix),
                   supergraph_relation(node, node_idx, node_embedding, subgraph_idx, subgraph_avg_dic, subsubgraph_distance_matrix)]
    return final_score



def calculate_score(supergraph_list_file, lower_text_file, last_level_file,
                    phrase_embedding_file, centroid_file, output_score_file):
    lower_text =[]
    supergraph_list = load_obj(supergraph_list_file)
    with open(lower_text_file, 'r', encoding='utf-8') as f_in:
        for text_i in f_in:
            text_i = text_i.strip('\n')
            lower_text.append(text_i)
    with open(last_level_file, 'r', encoding='utf-8') as f_last:
        last_level = json.load(f_last)
    phrase_embedding = load_obj(phrase_embedding_file)
    centroid_list = load_obj(centroid_file)
    output_score_list = []
    for item in tqdm(range(TOTAL_NUMBER)):
        score = {}
        # ipdb.set_trace()
        if len(supergraph_list[item]) == 0:
            output_score_list.append(score)
            continue
        subgraph_dic = supergraph_list[item]

        subgraph_embedding_dic = {}
        subgraph_avg_dic = {}
        subgraph_distance_matrix = {}
        for subgraph_idx in subgraph_dic:
            subgraph_embedding_dic[subgraph_idx] = []
            subgraph = subgraph_dic[subgraph_idx]
            for subgraph_item in subgraph:
                subgraph_embedding_dic[subgraph_idx].append(phrase_embedding[subgraph_item])
        for subgraph_idx in subgraph_dic:
            subgraph_avg_dic[subgraph_idx] = np.average(subgraph_embedding_dic[subgraph_idx])
        for subgraph_idx in subgraph_dic:
            temp_matrix = pdist(subgraph_embedding_dic[subgraph_idx], metric='cosine')
            subgraph_distance_matrix[subgraph_idx] = squareform(temp_matrix)

        influence_sphere_score = []
        for subgraph_idx in subgraph_dic:
            score[subgraph_idx] = {}
            subgraph = subgraph_dic[subgraph_idx]
            for subgraph_item in subgraph:
                score[subgraph_idx][subgraph_item] = supergraph_node_score(subgraph_item, subgraph[subgraph_item], phrase_embedding[subgraph_item], subgraph_idx, 
                        subgraph_avg_dic, subgraph_distance_matrix[subgraph_idx], lower_text[item], last_level[item], centroid_list)

        output_score_list.append(score)
    save_obj(output_score_list, output_score_file)


if __name__ == '__main__':
    supergraph_list_file = 'patent/claim/claim_graph/claim_supergraph_list.pkl'
    lower_text_file = 'example_data/example_claim/claim.txt'
    last_level_file = 'patent/abstract/abstract_rank/abstract_selected_phrase.json'
    output_score_file = 'patent/claim/claim_score/claim_influence_phrase_score.pkl'
    phrase_embedding_file = 'patent/claim/claim_embedding/cpc_title_abstract_claim_phrase_embedding.pkl'
    centroid_file = 'patent/abstract/abstract_clustering/abstract_influence_phrase_centroids.pkl'
    calculate_score(supergraph_list_file, lower_text_file, last_level_file,
                    phrase_embedding_file, centroid_file, output_score_file)



