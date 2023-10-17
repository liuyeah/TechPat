import pickle
import json
from tqdm import tqdm


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def ranked_phrase_text(ranked_phrase_file, output_file):
    ranked_phrase = load_obj(ranked_phrase_file)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx in tqdm(range(len(ranked_phrase))):
            item_dic = json.dumps(ranked_phrase[idx])
            f_out.write(item_dic)
            f_out.write('\n')




if __name__ == '__main__':
    ranked_phrase_file = 'patent/abstract/abstract_rank/ranked_abstract_influence_phrase_score.pkl'
    output_file = 'patent/abstract/abstract_rank/ranked_abstract_influence_phrase_score_text.json'
    ranked_phrase_text(ranked_phrase_file, output_file)
