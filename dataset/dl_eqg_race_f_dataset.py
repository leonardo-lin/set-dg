import json

from datasets import load_dataset

dataset = load_dataset("voidful/EQG-RACE-PLUS")


def process_dataset(data, question_type='factiod_questions'):
    jsonl_data = []
    for i in data:
        for q in i['questions']:
            if q['question_type'] == question_type:
                data_json = {'question': q['question'], 'passage': i['article'], 'options': q['options'],
                             'answer': q['answer']['answer_text'], 'answer_index': q['answer']['answer_index']}
                jsonl_data.append(data_json)
    return jsonl_data


def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


train_jsonl = process_dataset(dataset['train_all'])
write_jsonl('unfiltered/eqg_race_f/train.jsonl', train_jsonl)

test_jsonl_high = process_dataset(dataset['test_high'])
test_jsonl_middle = process_dataset(dataset['test_middle'])
test_jsonl = test_jsonl_high + test_jsonl_middle
write_jsonl('unfiltered/eqg_race_f/test.jsonl', test_jsonl)

dev_jsonl_high = process_dataset(dataset['dev_high'])
dev_jsonl_middle = process_dataset(dataset['dev_middle'])
dev_jsonl = dev_jsonl_high + dev_jsonl_middle
write_jsonl('unfiltered/eqg_race_f/dev.jsonl', dev_jsonl)
