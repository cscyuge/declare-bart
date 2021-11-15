import pickle
from pprint import pprint
import numpy as np
from transformers import BertTokenizer

with open('./data/test/cudics.pkl', 'rb') as f:
    cudics = pickle.load(f)

with open('./data/common.txt','r') as f:
    common = f.read().split('\n')

print(len(common))
common_special = []

for cudic in cudics:
    for key in cudic.keys():
        special = cudic[key].split(' ')
        for word in special:
            word = word.lower().strip().strip(',')
            if word not in common_special and not word.isdigit() and len(word)>1:
                common_special.append(word)
print(len(common_special))


input_ids=pickle.load(open("./data/input_ids",'rb'))
tags=pickle.load(open("./data/tags",'rb'))
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

for ids, labels in zip(input_ids,tags):
    for id, label in zip(ids,labels):
        if label==1:
            word = tokenizer.convert_ids_to_tokens([id])[0]
            print(word)
            if word not in common_special and not word.isdigit() and len(word) > 1:
                common_special.append(word)

common_special = common + common_special
common_special = list(set(common_special))
print(len(common_special))

with open('./data/common_special.txt','w', encoding='utf-8') as f:
    f.writelines([u+'\n' for u in common_special])



