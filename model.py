import pickle

import numpy as np
import torch
from importlib import import_module
from tqdm import tqdm
import copy
from utils.bleu_eval import count_score,count_hit,count_common
from utils.dataset import build_dataset, build_iterator, build_dataset_eval, build_iterator_eval
from transformers import BartForConditionalGeneration
from unit import unlikelihood_loss

PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'


def build(batch_size, cuda):

    x = import_module('config')
    pretrained_model = 'facebook/bart-base'
    config = x.Config(batch_size, pretrained_model)
    train_data = build_dataset('./data/train/src_ids.pkl', './data/train/src_masks.pkl',
                               './data/train/tar_ids.pkl','./data/train/tar_masks.pkl')
    test_data = build_dataset_eval('./data/test/src_ids.pkl', './data/test/src_masks.pkl',
                              './data/test/tar_txts.pkl', './data/test/cudics.pkl')
    val_data = build_dataset_eval('./data/test/src_ids.pkl', './data/test/src_masks.pkl',
                                   './data/test/tar_txts.pkl', './data/test/cudics.pkl')

    train_dataloader = build_iterator(train_data, config)
    val_dataloader = build_iterator_eval(val_data, config)
    test_dataloader = build_iterator_eval(test_data, config)

    model = BartForConditionalGeneration.from_pretrained(pretrained_model)
    model = model.to(config.device)
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.learning_rate
    )

    return model, optimizer, train_dataloader, val_dataloader, test_dataloader, config


def eval_set(model, dataloader, config):
    model.eval()
    results = []
    references = []
    for i, (batch_src, tars, cudics) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            outputs = model.generate(input_ids=batch_src[0], attention_mask=batch_src[1], max_length=config.pad_size)
            outputs = config.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results += outputs
            references += tars

    results = [u.lower().replace('[mask]','').
                            replace('[ mask]','').
                            replace('[mask ]','').
                            replace('[m ask]', '').
                            replace('[mas k]', '').
                            replace('  ',' ') for u in results]
    references = [[v.lower().replace('[mask]','').replace('  ',' ') for v in u] for u in references]
    hyps = results[:]
    refs = references[:]
    with open('./data/test_dic.pkl', 'rb') as f:
        test_dics = pickle.load(f)
    bleu = count_score(hyps, refs)
    hit = count_hit(results, test_dics)
    common = count_common(results)
    print('BLEU:{}, HIT:{}, COMMON:{}'.format(bleu, hit, common))
    model.train()
    return bleu, hit, common, results, references


def train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config):
    #training steps
    max_bleu = -99999
    save_file = {}
    with open('./data/weight_vector.pkl','rb') as f:
        weight_vector = pickle.load(f)
    weight_vector = torch.LongTensor(weight_vector).to(config.device)

    for e in range(config.num_epochs):
        model.train()
        for i, (batch_src, batch_tar) in tqdm(enumerate(train_dataloader)):
            model_outputs = model(input_ids=batch_src[0], attention_mask=batch_src[1], labels=batch_tar[0])
            loss = model_outputs.loss

            ul_loss = unlikelihood_loss(batch_tar[0], model_outputs.logits, weight_vector)
            ul_loss_weighted = ul_loss * 100
            loss += ul_loss_weighted

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                print('train loss:%f' %loss.item())


        #validation steps
        if e >= 0:
            bleu, hit, common, results, references = eval_set(model, val_dataloader, config)
            print(results[0:5])
            print('BLEU:%f' %(bleu))
            if bleu > max_bleu:
                max_bleu = bleu
                save_file['epoch'] = e + 1
                save_file['para'] = model.state_dict()
                save_file['best_bleu'] = bleu
                torch.save(save_file, './cache/best_save.data')
            if bleu < max_bleu - 0.6:
                print('Early Stop')
                break
            print(save_file['epoch'] - 1)


    save_file_best = torch.load('./cache/best_save.data')
    print('Train finished')
    print('Best Val BLEU:%f' %(save_file_best['best_bleu']))
    model.load_state_dict(save_file_best['para'])
    bleu, hit, common, results, references = eval_set(model, test_dataloader, config)
    with open('./result/best_save_bert.out.txt', 'w', encoding="utf-8") as f:
        f.writelines([x + '\n' for x in results])
    with open('./result/results.pkl','wb') as f:
        pickle.dump(results, f)

    print('Test BLEU:%f' % (bleu))
    print('Test HIT:%f' %(hit))
    print('Test COMMON:%f' % (common))


def main():
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(32, True)
    train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config)
    print('finish')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
