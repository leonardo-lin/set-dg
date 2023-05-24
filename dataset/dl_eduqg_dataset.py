import json
import os

import nlp2


def download_and_read_data(file_url, file_path):
    # Uncomment the following line if you want to download the files
    # get folder from file path
    if not nlp2.is_file_exist(file_path):
        file_folder = os.path.dirname(file_path)
        nlp2.download_file(file_url, file_folder)
    return nlp2.read_json(file_path)


def convert_data_format(data):
    new_data = {}

    new_data['question'] = data['question']['normal_format']
    new_data['passage'] = data['hl_context'].replace("<hl>", "").strip()
    new_data['options'] = data['question']['question_choices']
    new_data['answer'] = data['question']['question_choices'][data['answer']['ans_choice']]
    new_data['answer_index'] = data['answer']['ans_choice']

    return new_data


def process_data(data_list):
    processed_data = []
    for data in data_list:
        for question in data['questions']:
            processed_data.append(convert_data_format(question))
    return processed_data


def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


train_data_url = 'https://huggingface.co/datasets/voidful/EduQG/resolve/main/qg_train_v0.json'
valid_data_url = 'https://huggingface.co/datasets/voidful/EduQG/raw/main/qg_valid_v0.json'

train_data_path = './dummy/qg_train_v0.json'
valid_data_path = './dummy/qg_valid_v0.json'

train_data = download_and_read_data(train_data_url, train_data_path)
valid_data = download_and_read_data(valid_data_url, valid_data_path)

processed_train_data = process_data(train_data)
processed_valid_data = process_data(valid_data)

# remove_dummy_folder
import shutil

shutil.rmtree('./dummy/')

write_jsonl('./unfiltered/eduqg/train.jsonl', processed_train_data)
write_jsonl('./unfiltered/eduqg/valid.jsonl', processed_valid_data)
