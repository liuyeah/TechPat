import json


def cpc2list(cpc_phrase_file, output_file):
    output = []
    with open(cpc_phrase_file, 'r', encoding='utf-8') as f_in:
        for item in f_in:
            item = item.strip('\n')
            output.append(item)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(output, f_out, indent=True)


if __name__ == '__main__':
    cpc_phrase_file = 'example_data/example_cpc/cpc_phrase.txt'
    output_file = 'patent/cpc/cpc_phrase_list/cpc_phrase_list.json'
    cpc2list(cpc_phrase_file, output_file)
