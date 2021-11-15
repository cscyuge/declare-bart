PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'
from tqdm import tqdm
import re
import nltk
import pickle
import numpy as np
import torch

def build_dataset(src_ids_path, src_masks_path, tar_ids_path, tar_masks_path):
    token_ids_srcs = pickle.load(open(src_ids_path, 'rb'))
    mask_srcs = pickle.load(open(src_masks_path, 'rb'))

    token_ids_tars = pickle.load(open(tar_ids_path, 'rb'))
    mask_tars = pickle.load(open(tar_masks_path,'rb'))

    dataset = []
    for token_ids_src, mask_src, token_ids_tar, mask_tar in zip(token_ids_srcs, mask_srcs, token_ids_tars, mask_tars):
        dataset.append((token_ids_src, mask_src, token_ids_tar, mask_tar))
    return dataset

def build_dataset_eval(src_ids_path, src_masks_path, tar_txts_path, cudics_path):
    token_ids_srcs = pickle.load(open(src_ids_path, 'rb'))
    mask_srcs = pickle.load(open(src_masks_path, 'rb'))

    tar_txts = pickle.load(open(tar_txts_path, 'rb'))
    cudics = pickle.load(open(cudics_path, 'rb'))
    dataset = []
    for token_ids_src, mask_src, tars, cudic in zip(token_ids_srcs, mask_srcs, tar_txts, cudics):
        dataset.append((token_ids_src, mask_src, tars, cudic))
    return dataset


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x_src = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        mask_src = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        x_tar = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask_tar = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        return (x_src, mask_src), (x_tar, mask_tar)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


class DatasetIteraterEval(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x_src = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        mask_src = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        x_tar = [_[2] for _ in datas]
        dic_tar = [_[3] for _ in datas]
        return (x_src, mask_src), x_tar, dic_tar

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def build_iterator_eval(dataset, config):
    iter = DatasetIteraterEval(dataset, config.batch_size_eval, config.device)
    return iter


