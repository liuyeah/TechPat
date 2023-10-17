import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
from tqdm import tqdm
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)



class superNeGraph(object):
    def __init__(self, supergraph_dic, value_dic, similarity_matrix_dic, subgraph_avg_embedding, phrase_embedding):
        self.supergraph_dic = supergraph_dic
        self.value_dic = value_dic
        self.similarity_matrix_dic = similarity_matrix_dic
        self.subgraph_avg_embedding = subgraph_avg_embedding
        self.subgraph_avg_pr_dic = {}
        self.phrase_embedding = phrase_embedding
        self.pr_dic = {}
        node_sum = 0
        for subgraph_idx in supergraph_dic:
            node_sum = node_sum + len(supergraph_dic[subgraph_idx])
        for subgraph_idx in supergraph_dic:
            subgraph = supergraph_dic[subgraph_idx]
            self.pr_dic[subgraph_idx] = []
            for item in subgraph:
                self.pr_dic[subgraph_idx].append(1/node_sum)
    

    def Penalty(self, v_j, v_i):
        overlapping = 0
        i_list = v_i.strip(' ')
        j_list = v_j.strip(' ')
        for item in i_list:
            if item in j_list:
                overlapping = overlapping + 1
        return 1 - overlapping/len(j_list)
    
    
    def update_single_centroid(self, subgraph, subgraph_pr, avg_embedding):
        item_similarity_dic = {}
        item_similarity_sum = 0
        avg_pr = 0
        for item in subgraph:
            item_idx = subgraph[item]
            item_embedding = self.phrase_embedding[item]
            item_similarity_dic[item] = 1 - cosine(item_embedding, avg_embedding)
            item_similarity_sum = item_similarity_sum + item_similarity_dic[item]
        for item in subgraph:
            avg_pr = avg_pr + subgraph_pr[item_idx] * item_similarity_dic[item] / item_similarity_sum
        return avg_pr
    
    def update_supergraph_centroids(self):
        for subgraph_idx in self.supergraph_dic:
            subgraph = self.supergraph_dic[subgraph_idx]
            subgraph_pr = self.pr_dic[subgraph_idx]
            avg_embedding = self.subgraph_avg_embedding[subgraph_idx]
            self.subgraph_avg_pr_dic[subgraph_idx] = self.update_single_centroid(subgraph, subgraph_pr, avg_embedding)


    def calculate_pr(self, d, beta):
        temp_pr_dic = self.pr_dic.copy()
        # global ne-rank
        self.update_supergraph_centroids()
        subgraph_order = {}
        avg_embedding_list = []
        avg_sum_dic = {}
        for subgraph_idx in self.supergraph_dic:
            subgraph_order[subgraph_idx] = len(subgraph_order)
            avg_embedding_list.append(self.subgraph_avg_embedding[subgraph_idx])
        temp_matrix = pdist(avg_embedding_list, metric='cosine')
        avg_distance_matrix = squareform(temp_matrix)
        avg_similarity_matrix = np.ones((len(avg_distance_matrix), len(avg_distance_matrix)), dtype='float64') - avg_distance_matrix
        for subgraph_idx in subgraph_order:
            avg_sum_dic[subgraph_idx] = np.sum(avg_similarity_matrix[subgraph_order[subgraph_idx]])
        
        for subgraph_idx in self.supergraph_dic:
            subgraph = self.supergraph_dic[subgraph_idx]
            subgraph_value = self.value_dic[subgraph_idx]
            avg_temp_sum = 0
            for avg_idx in self.subgraph_avg_embedding:
                if avg_idx == subgraph_idx:
                    continue
                avg_temp_sum = avg_temp_sum + avg_similarity_matrix[subgraph_order[subgraph_idx]][subgraph_order[avg_idx]] / avg_sum_dic[avg_idx] * self.subgraph_avg_pr_dic[avg_idx]
            for item in subgraph:
                item_idx = subgraph[item]
                self.pr_dic[subgraph_idx][item_idx] = d * (1-beta) * subgraph_value[item] * avg_temp_sum * (1-cosine(self.phrase_embedding[item], self.subgraph_avg_embedding[subgraph_idx]))


        # local ne-rank
        for subgraph_idx in self.supergraph_dic:
            subgraph = self.supergraph_dic[subgraph_idx]
            subgraph_value = self.value_dic[subgraph_idx]
            subgraph_similarity_matrix = self.similarity_matrix_dic[subgraph_idx]
            temp_pr = temp_pr_dic[subgraph_idx]
            modify_sum = []

            penalty_item_dic = {}
            for item in subgraph:
                penalty_item_dic[len(penalty_item_dic)] = item

            for modify_idx in range(len(subgraph_similarity_matrix)):
                modify_sum.append(np.sum(subgraph_similarity_matrix[modify_idx]))
            # local ne-rank
            for item in subgraph:
                item_idx = subgraph[item]
                item_bias = (1 - d) * subgraph_value[item]
                modify_value = 0.0
                for temp_idx in range(len(subgraph_similarity_matrix[item_idx])):
                    if item_idx == temp_idx or modify_sum[temp_idx] == 0.0:
                        continue
                    modify_value = modify_value + subgraph_similarity_matrix[item_idx][temp_idx]/modify_sum[temp_idx] * \
                                temp_pr[temp_idx] * self.Penalty(penalty_item_dic[temp_idx], penalty_item_dic[item_idx])
                if modify_value == 0.0:
                    self.pr_dic[subgraph_idx][item_idx] = self.pr_dic[subgraph_idx][item_idx] + item_bias
                    continue
                self.pr_dic[subgraph_idx][item_idx] = self.pr_dic[subgraph_idx][item_idx] + item_bias + d * beta * subgraph_value[item] * modify_value
    
    
    def calculate_pr_times(self, d, beta, times):
        for i in range(times):
            self.calculate_pr(d, beta)

    def calculate_pr_converge(self, d, beta, threshold):
        change_dic = {}
        for subgraph_idx in self.supergraph_dic:
            subgraph = self.supergraph_dic[subgraph_idx]
            change_dic[subgraph_idx] = np.array([100.0] * len(subgraph))
        count_number = 0
        while 1:
            count_number = count_number + 1
            out_signal = 0
            old_pr_dic = self.pr_dic.copy()
            self.calculate_pr(d, beta)
            for subgraph_idx in change_dic:
                for i in range(len(change_dic[subgraph_idx])):
                    change_dic[subgraph_idx][i] = abs(self.pr_dic[subgraph_idx][i] - old_pr_dic[subgraph_idx][i])
                if (change_dic[subgraph_idx] > threshold).any():
                    out_signal = 1
            if out_signal == 0 or count_number >= 100:
                break


    def get_final_pr_score_dic(self):
        temp_pr_score = {}
        final_pr_score = {}
        for subgraph_idx in self.supergraph_dic:
            subgraph = self.supergraph_dic[subgraph_idx]
            for item in subgraph:
                item_idx = subgraph[item]
                temp_pr_score[item] = self.pr_dic[subgraph_idx][item_idx]
        final_pr_score_list = sorted(temp_pr_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        for i in final_pr_score_list:
            final_pr_score[i[0]] = i[1]
        return final_pr_score



def super_test(supergraph_list_file, phrase_embedding_file, graph_value_list_file, output_final_pr_file):
    supergraph_list = load_obj(supergraph_list_file)
    phrase_embedding = load_obj(phrase_embedding_file)
    graph_value_list = load_obj(graph_value_list_file)
    final_phrase = []
    for idx in tqdm(range(len(supergraph_list))):
        if len(supergraph_list[idx]) == 0:
            final_phrase.append({})
            continue
        supergraph_dic = supergraph_list[idx]
        subgraph_embedding_dic = {}
        subgraph_avg_dic = {}
        subgraph_distance_matrix = {}
        subgraph_similarity_matrix = {}
        for subgraph_idx in supergraph_dic:
            subgraph_embedding_dic[subgraph_idx] = []
            subgraph = supergraph_dic[subgraph_idx]
            for subgraph_item in subgraph:
                subgraph_embedding_dic[subgraph_idx].append(phrase_embedding[subgraph_item])

        for subgraph_idx in supergraph_dic:
            subgraph_avg_dic[subgraph_idx] = np.average(subgraph_embedding_dic[subgraph_idx], axis=0)

        for subgraph_idx in supergraph_dic:
            temp_matrix = pdist(subgraph_embedding_dic[subgraph_idx], metric='cosine')
            subgraph_distance_matrix[subgraph_idx] = squareform(temp_matrix)
            subgraph_similarity_matrix[subgraph_idx] = np.ones((len(subgraph_distance_matrix[subgraph_idx]), len(subgraph_distance_matrix[subgraph_idx])), dtype='float64') - subgraph_distance_matrix[subgraph_idx]
        
        supergraph_score = superNeGraph(supergraph_dic, graph_value_list[idx], subgraph_similarity_matrix, subgraph_avg_dic, phrase_embedding)
        supergraph_score.calculate_pr_converge(0.85, 0.5, 0.0001)
        final_phrase.append(supergraph_score.get_final_pr_score_dic())
    save_obj(final_phrase, output_final_pr_file)



if __name__ == '__main__':
    supergraph_list_file = 'patent/title/title_graph/title_supergraph_list.pkl'
    phrase_embedding_file = 'patent/title/title_embedding/cpc_title_phrase_embedding.pkl'
    graph_value_list_file = 'patent/title/title_score/title_influence_phrase_list_normalized_score.pkl'
    test_file = 'patent/title/title_rank/ranked_title_influence_phrase_score.pkl'
    super_test(supergraph_list_file, phrase_embedding_file, graph_value_list_file, test_file)

