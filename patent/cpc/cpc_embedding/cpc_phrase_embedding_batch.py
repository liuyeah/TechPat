import numpy as np
import gensim
import json
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import ipdb
import pickle
import logging
logging.basicConfig(level=logging.DEBUG)
import os
from tqdm import tqdm
from bert_serving.client import BertClient
EMBEDDING_SIZE = int(os.environ.get('EMBEDDING_SIZE'))
EMBEDDING_BATCH = int(os.environ.get('EMBEDDING_BATCH'))

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def cut_list(lists, cut_len):
    res_data = []
    if len(lists) > cut_len:
        for i in range(int(len(lists) / cut_len)):
            cut_a = lists[cut_len * i:cut_len * (i + 1)]
            res_data.append(cut_a)

        last_data = lists[int(len(lists) / cut_len) * cut_len:]
        if last_data:
            res_data.append(last_data)
    else:
        res_data.append(lists)

    return res_data


def batch_bert_phrase_embedding(cpc_phrase_file, output_file, batch_size=EMBEDDING_BATCH):
    bc = BertClient(port=7777, port_out=7778)
    phrase_embedding = {}
    phrase_embedding_keys = []
    with open(cpc_phrase_file, 'r', encoding='utf-8') as f_in:
        for item in f_in:
            item = item.strip('\n')
            temp_phrase = item.lower()
            if temp_phrase != '' and temp_phrase not in phrase_embedding_keys:
                phrase_embedding_keys.append(temp_phrase)
    
    batched_keys = cut_list(phrase_embedding_keys, batch_size)

    for batch in tqdm(batched_keys, total=len(batched_keys)):
        batch_embedding = bc.encode(batch)
        for idx in range(len(batch)):
            phrase_embedding[batch[idx]] = batch_embedding[idx]

    save_obj(phrase_embedding, output_file)



if __name__ == '__main__':
    cpc_phrase_file = 'example_data/example_cpc/cpc_phrase.txt'
    output_file = 'patent/cpc/cpc_embedding/cpc_phrase_embedding.pkl'
    batch_bert_phrase_embedding(cpc_phrase_file, output_file)

