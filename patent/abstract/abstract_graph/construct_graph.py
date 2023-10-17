import numpy as np
import pickle
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine, pdist, squareform
import hdbscan
import os
TOTAL_NUMBER = int(os.environ.get('TOTAL_NUMBER'))
ABSTRACT_SUPERGRAPH_CLUSTER = int(os.environ.get('ABSTRACT_SUPERGRAPH_CLUSTER'))


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def build_graph(super_sapn_file, phrase_embedding_file, output_supergraph_file):
    supergraph_list = []
    with open(super_sapn_file, 'r', encoding='utf-8') as f_in:
        super_span = json.load(f_in)
    phrase_embedding = load_obj(phrase_embedding_file)
    for spans in tqdm(super_span, total=TOTAL_NUMBER):
        # construct the basic graph
        supergraph = {}

        graph = {}
        graph_embedding = []
        graph_number = 0
        temp_list = spans['superspan']
        count = 0
        for j in temp_list:
            temp_node = j.lower()
            if temp_node != '' and temp_node not in graph:
                if temp_node in phrase_embedding:
                    if not (phrase_embedding[temp_node] == 0.0).all():
                        graph[temp_node] = count
                        count = count + 1
                        graph_embedding.append(phrase_embedding[temp_node])
                    else:
                        print("zero embedding phrase: " + temp_node)
                else:
                    print('cannot find graph : ' + str(graph_number) + ', node: ' + temp_node)
            
        if len(graph) > 1:
            cluster_er = hdbscan.HDBSCAN(min_cluster_size=ABSTRACT_SUPERGRAPH_CLUSTER)
            cluster_labels = cluster_er.fit_predict(np.array(graph_embedding))
            for item, idx in zip(graph, cluster_labels):
                if idx not in supergraph:
                    supergraph[idx] = {}
                supergraph[idx][item] = len(supergraph[idx])
        elif len(graph) == 1:
            for item in graph:
                supergraph[0] = {item: 0}

        graph_number = graph_number + 1
        supergraph_list.append(supergraph)

    save_obj(supergraph_list, output_supergraph_file)



if __name__ == '__main__':
    super_sapn_file = 'patent/abstract/abstract_candidate/abstract_candidate_synthesis.json'
    phrase_embedding_file = 'patent/abstract/abstract_embedding/cpc_title_abstract_phrase_embedding.pkl'
    output_supergraph_file = 'patent/abstract/abstract_graph/abstract_supergraph_list.pkl'
    build_graph(super_sapn_file, phrase_embedding_file, output_supergraph_file)




