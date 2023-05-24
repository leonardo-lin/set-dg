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
    new_data = {
        'question': data['question'],
        'passage': data['context'],
        'options': data['answers'],
        'answer': data['answers'][data['label']],
        'answer_index': data['label']
    }
    return new_data


def process_data(data_list):
    processed_data = []
    for data in data_list:
        processed_data.append(convert_data_format(data))
    return processed_data


def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


train_data_url = 'https://huggingface.co/datasets/voidful/ReClor/resolve/main/train.json'
valid_data_url = 'https://huggingface.co/datasets/voidful/ReClor/resolve/main/val.json'

train_data_path = './dummy/train.json'
valid_data_path = './dummy/val.json'

train_data = download_and_read_data(train_data_url, train_data_path)
valid_data = download_and_read_data(valid_data_url, valid_data_path)

processed_train_data = process_data(train_data)
processed_valid_data = process_data(valid_data)

# remove_dummy_folder
import shutil

shutil.rmtree('./dummy/')

write_jsonl('./unfiltered/reclor/train.jsonl', processed_train_data)
write_jsonl('./unfiltered/reclor/valid.jsonl', processed_valid_data)
