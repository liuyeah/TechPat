import json
from tqdm import tqdm
import os
TOTAL_NUMBER = int(os.environ.get('TOTAL_NUMBER'))


def candidate_synthesis(original_candidate_file, output_file):
    with open(original_candidate_file, 'r', encoding='utf-8') as f_in:
        candidate_phrase = []
        for item in tqdm(f_in, total=TOTAL_NUMBER):
            title_candidate = {}
            item = item.strip('\n')
            superspan = json.loads(item)
            temp_plain = []
            temp_super = []
            temp_plain_st_ed = []
            temp_super_st_ed = []
            for span in superspan:
                if span['tag'] == "plain":
                    if (span['st'], span['ed']) not in temp_plain_st_ed:
                        temp_plain.append(span['text'])
                        temp_plain_st_ed.append((span['st'], span['ed']))
                elif span['tag'] == "superspan":
                    for phrase_j in span['spans']:
                        if (phrase_j['st'], phrase_j['ed']) not in temp_super_st_ed:
                            temp_super.append(phrase_j['text'])
                            temp_super_st_ed.append((phrase_j['st'], phrase_j['ed']))
            title_candidate['plain'] = temp_plain
            title_candidate['superspan'] = temp_super
            candidate_phrase.append(title_candidate)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(candidate_phrase, f_out, indent=True)


if __name__ == '__main__':
    original_candidate_file = 'example_data/example_claim/claim.txt_superspan_sequence.json'
    output_file = 'patent/claim/claim_candidate/claim_candidate_synthesis.json'
    candidate_synthesis(original_candidate_file, output_file)
