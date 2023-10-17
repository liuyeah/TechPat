from nltk.tokenize import sent_tokenize
import json
import re

title_path = 'example_data/example_title/title.txt'
title_list_path = 'patent/title/title_rank/ranked_title_influence_phrase_score_text.json'

title_result_path = 'patent/title/title_rank/title_selected_phrase.json'

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


with open(title_result_path, 'w', encoding='utf-8') as f_out:
    json.dump(title_result, f_out, indent=True)
