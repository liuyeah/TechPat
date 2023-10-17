from nltk.tokenize import sent_tokenize
import json
import re

claim_path = 'example_data/example_claim/claim.txt'
claim_list_path = 'patent/claim/claim_rank/ranked_claim_influence_phrase_score_text.json'

claim_result_path = 'patent/claim/claim_rank/claim_selected_phrase.json'

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


with open(claim_result_path, 'w', encoding='utf-8') as f_out:
    json.dump(claim_result, f_out, indent=True)