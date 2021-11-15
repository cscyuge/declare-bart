import pickle
from transformers import BartTokenizer
from tqdm import tqdm
import re
import nltk

bert_model = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(bert_model)
CLS = tokenizer.cls_token
SEP = tokenizer.sep_token


def generate_train(data, pad_size=64):
    data = data.lower()
    data = data.replace('[mask]', '[MASK]')
    data = data.split('\n\n')
    src_ids = []
    tar_ids = []
    src_masks = []
    tar_masks = []
    for lines in tqdm(data):
        lines = lines.split('\n')
        src = lines[0]
        tar = lines[1]
        token_src = tokenizer.tokenize(src)
        token_src = [CLS] + token_src + [SEP]
        token_tar = tokenizer.tokenize(tar)
        token_tar = [CLS] + token_tar + [SEP]

        mask_src = []
        token_ids_src = tokenizer.convert_tokens_to_ids(token_src)

        if pad_size:
            if len(token_src) < pad_size:
                mask_src = [1] * len(token_ids_src) + [0] * (pad_size - len(token_src))
                token_ids_src += ([0] * (pad_size - len(token_src)))
            else:
                mask_src = [1] * pad_size
                token_ids_src = token_ids_src[:pad_size]

        mask_tar = []
        token_ids_tar = tokenizer.convert_tokens_to_ids(token_tar)

        if pad_size:
            if len(token_tar) < pad_size:
                mask_tar = [1] * len(token_ids_tar) + [0] * (pad_size - len(token_tar))
                token_ids_tar += ([0] * (pad_size - len(token_tar)))
            else:
                mask_tar = [1] * pad_size
                token_ids_tar = token_ids_tar[:pad_size]

        src_ids.append(token_ids_src)
        src_masks.append(mask_src)
        tar_ids.append(token_ids_tar)
        tar_masks.append(mask_tar)

    return src_ids, src_masks, tar_ids, tar_masks

def generator_eval(data, pad_size=64):
    data = data.lower()
    data = data.replace('[mask]', '[MASK]')
    data = data.split('\n\n')
    src_ids = []
    src_masks = []
    tar_txts = []
    cudics = []
    for lines in tqdm(data):
        lines = lines.split('\n')
        cudic = {}
        tars = []
        for sid, sentence in enumerate(lines):
            if sid == 0:
                src = lines[0]
                token_src = tokenizer.tokenize(src)
                token_src = [CLS] + token_src + [SEP]
            else:
                sentence = sentence[2:]
                text = re.sub('\[[^\[\]]*\]', '', sentence)
                pairs = re.findall('[^\[\] ]+\[[^\[\]]+\]', sentence)
                for pair in pairs:
                    pair = re.split('[\[\]]', pair)
                    cudic[pair[0]] = pair[1]
                words = nltk.word_tokenize(text)
                for wid, word in enumerate(words):
                    if word in cudic.keys():
                        words[wid] = cudic[word]
                new_text = ''
                for word in words:
                    new_text += word
                    new_text += ' '
                tars.append(new_text)

        mask_src = []
        token_ids_src = tokenizer.convert_tokens_to_ids(token_src)

        if pad_size:
            if len(token_src) < pad_size:
                mask_src = [1] * len(token_ids_src) + [0] * (pad_size - len(token_src))
                token_ids_src += ([0] * (pad_size - len(token_src)))
            else:
                mask_src = [1] * pad_size
                token_ids_src = token_ids_src[:pad_size]

        src_ids.append(token_ids_src)
        src_masks.append(mask_src)
        tar_txts.append(tars)
        cudics.append(cudic)

    return src_ids, src_masks, tar_txts, cudics


def main():
    train_data = open('./data/train(12809).txt', 'r', encoding='utf-8').read()
    src_ids, src_masks, tar_ids, tar_masks = generate_train(train_data, pad_size=64)
    with open('./data/train/src_ids.pkl','wb') as f:
        pickle.dump(src_ids, f)
    with open('./data/train/src_masks.pkl', 'wb') as f:
        pickle.dump(src_masks, f)
    with open('./data/train/tar_ids.pkl','wb') as f:
        pickle.dump(tar_ids, f)
    with open('./data/train/tar_masks.pkl', 'wb') as f:
        pickle.dump(tar_masks, f)

    eval_data = open('./data/test(2030).txt', 'r', encoding='utf-8').read()
    src_ids, src_masks, tar_txts, cudics = generator_eval(eval_data, pad_size=64)

    with open('./data/test/src_ids.pkl', 'wb') as f:
        pickle.dump(src_ids, f)
    with open('./data/test/src_masks.pkl', 'wb') as f:
        pickle.dump(src_masks, f)
    with open('./data/test/tar_txts.pkl', 'wb') as f:
        pickle.dump(tar_txts, f)
    with open('./data/test/cudics.pkl', 'wb') as f:
        pickle.dump(cudics, f)


if __name__ == '__main__':
    main()
