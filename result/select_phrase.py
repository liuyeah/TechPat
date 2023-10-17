from nltk.tokenize import sent_tokenize
import json
import re

title_path = 'example_data/example_title/title.txt'
abstract_path = 'example_data/example_abstract/abstract.txt'
claim_path = 'example_data/example_claim/claim.txt'

title_list_path = 'patent/title/title_rank/ranked_title_influence_phrase_score_text.json'
abstract_list_path = 'patent/abstract/abstract_rank/ranked_abstract_influence_phrase_score_text.json'
claim_list_path = 'patent/claim/claim_rank/ranked_claim_influence_phrase_score_text.json'

final_result_path = 'result/final_output.json'

title_sen = []
title_result = []
count = 0
with open(title_path, 'r', encoding='utf-8') as f_title:
    for item in f_title:
        item = item.strip('\n')
        temp_len = len(sent_tokenize(item))
        title_sen.append(temp_len)
with open(title_list_path, 'r', encoding='utf-8') as f_title:
    for item in f_title:
        item = item.strip('\n')
        item_dic = json.loads(item)
        temp_result = []
        for k in item_dic:
            temp_result.append(k)
        temp_result = temp_result[:2*title_sen[count]]
        title_result.append(temp_result)
        count = count + 1

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

claim_sen = []
claim_result = []
count = 0
with open(claim_path, 'r', encoding='utf-8') as f_claim:
    for item in f_claim:
        item = item.strip('\n')
        b = sent_tokenize(item)
        neg = 0
        for temp_item in b:
            num = re.sub(r'\d', "", temp_item.strip('. '))
            if num == '':
                neg = neg + 1
        temp_len = len(b) - neg
        claim_sen.append(temp_len)
with open(claim_list_path, 'r', encoding='utf-8') as f_claim:
    for item in f_claim:
        item = item.strip('\n')
        item_dic = json.loads(item)
        temp_result = []
        for k in item_dic:
            temp_result.append(k)
        temp_result = temp_result[:claim_sen[count]]
        claim_result.append(temp_result)
        count = count + 1

final_result = []
for i in range(len(title_result)):
    temp_final = {'title': title_result[i], 'abstract': abstract_result[i], 'claim': claim_result[i]}
    final_result.append(temp_final)

with open(final_result_path, 'w', encoding='utf-8') as f_out:
    json.dump(final_result, f_out, indent=True)
