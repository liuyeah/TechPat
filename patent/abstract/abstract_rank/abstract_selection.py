from nltk.tokenize import sent_tokenize
import json
import re

abstract_path = 'example_data/example_abstract/abstract.txt'
abstract_list_path = 'patent/abstract/abstract_rank/ranked_abstract_influence_phrase_score_text.json'

abstract_result_path = 'patent/abstract/abstract_rank/abstract_selected_phrase.json'

abstract_sen = []
abstract_result = []
count = 0
with open(abstract_path, 'r', encoding='utf-8') as f_abstract:
    for item in f_abstract:
        item = item.strip('\n')
        temp_len = len(sent_tokenize(item))
        abstract_sen.append(temp_len)
with open(abstract_list_path, 'r', encoding='utf-8') as f_abstract:
    for item in f_abstract:
        item = item.strip('\n')
        item_dic = json.loads(item)
        temp_result = []
        for k in item_dic:
            temp_result.append(k)
        temp_result = temp_result[:2*abstract_sen[count]]
        abstract_result.append(temp_result)
        count = count + 1


with open(abstract_result_path, 'w', encoding='utf-8') as f_out:
    json.dump(abstract_result, f_out, indent=True)