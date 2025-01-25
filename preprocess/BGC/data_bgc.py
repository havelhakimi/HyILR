from collections import defaultdict
from transformers import AutoTokenizer
import torch
import json
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    # string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " ", string)
    # string = re.sub(r"\.", " ", string)
    # string = re.sub(r"\"", " ", string)
    # string = re.sub(r"!", " ", string)
    # string = re.sub(r"\(", " ", string)
    # string = re.sub(r"\)", " ", string)
    # string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def process_file(file_path, split_key, tokenizer, source, labels, split, start_idx):
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line = json.loads(line)
            token=clean_str(line['token'].strip().lower())
            source.append(tokenizer.encode(token, truncation=True))
            labels.append(line['label'])
            split[split_key].append(start_idx + idx)
    return start_idx + len(split[split_key])



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    hiera = defaultdict(set)
    label_dict={}
    with open('bgc.taxonomy', 'r') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')
        hiera.pop(-1)
    value_dict = {i: tokenizer.encode(v.lower(), add_special_tokens=False) for v, i in label_dict.items()}
    #print(len(value_dict))
    torch.save(value_dict, 'bert_value_dict.pt')
    torch.save(hiera, 'slot.pt')

    # Initialize variables
    source = []
    labels = []
    split = {'train': [], 'val': [], 'test': []}
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Process each dataset
    idx = 0
    idx = process_file('train_data.jsonl', 'train', tokenizer, source, labels, split, idx)
    idx = process_file('dev_data.jsonl', 'val', tokenizer, source, labels, split, idx)
    idx = process_file('test_data.jsonl', 'test', tokenizer, source, labels, split, idx)

    labels_idx=[ [label_dict[lbl] for lbl in  label] for label in labels]

    with open('tok.txt', 'w') as f:
        for s in source:
            f.writelines(' '.join(map(lambda x: str(x), s)) + '\n')
    with open('Y.txt', 'w') as f:
        for s in labels_idx:
            one_hot = [0] * len(label_dict)
            for i in s:
                one_hot[i] = 1
            f.writelines(' '.join(map(lambda x: str(x), one_hot)) + '\n')
    
    print('Files created !')






