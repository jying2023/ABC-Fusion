import os
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, TensorDataset
from transformers.models.bert import BertTokenizer
import pdb

class ConfuseDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len=200, use_cache=True, candidates_num=5, subset='train'):
        cache_file = 'data/' + subset +'_' + str(candidates_num) + '_csc_dataset.pt'
        if use_cache and os.path.exists(cache_file):
            dataset = torch.load(cache_file)
            self.all_src_ids = dataset["src_ids"]
            self.all_attention_mask= dataset["attention_mask"]
            self.all_segment_ids = dataset["segment_ids"]
            self.all_tgt_ids = dataset["tgt_ids"]
            self.all_candidates_ids = dataset["candidates_ids"]

        else:
            lines = open(data_path, 'r', encoding='utf-8').readlines()
            example_list = []
            for line in lines:
                line = line.strip()
                pairs = line.split(" ")
                src = ' '.join(list(pairs[0]))
                tgt = ' '.join(list(pairs[1]))
                example_list.append((src, tgt))

            confusion_set = defaultdict(list)
            lines = open('data/confusion.txt', 'r', encoding='utf-8').readlines()
            for line in lines:
                 pairs = line.strip().split('|')
                if len(pairs[1]) + len(pairs[2]) <= candidates_num:
                    candidates = []
                    for c in pairs[1]:
                        tc = tokenizer.tokenize(c)[0]
                        if tc != tokenizer.unk_token:
                            candidates.append(tc)
                    for c in pairs[2]:
                        tc = tokenizer.tokenize(c)[0]
                        if tc != tokenizer.unk_token:
                            candidates.append(tc)
                    confusion_set[pairs[0]] = candidates
                else:
                    a = round(candidates_num*0.5)
                    if len(pairs[1]) < a:
                        candidates = []
                        for c in pairs[1]:
                            tc = tokenizer.tokenize(c)[0]
                            if tc != tokenizer.unk_token:
                                candidates.append(tc)
                        for c in random.sample(pairs[2], candidates_num - len(pairs[1])):
                            tc = tokenizer.tokenize(c)[0]
                            if tc != tokenizer.unk_token:
                                candidates.append(tc)
                        confusion_set[pairs[0]] = candidates
                    if len(pairs[2]) < a:
                        candidates = []
                        for c in pairs[2]:
                            tc = tokenizer.tokenize(c)[0]
                            if tc != tokenizer.unk_token:
                                candidates.append(tc)
                        for c in random.sample(pairs[1], candidates_num - len(pairs[2])):
                            tc = tokenizer.tokenize(c)[0]
                            if tc != tokenizer.unk_token:
                                candidates.append(tc)
                        confusion_set[pairs[0]] = candidates
                    if len(pairs[1]) >= a and len(pairs[2]) >= a:
                        candidates = []
                        for c in random.sample(pairs[1], a):
                            tc = tokenizer.tokenize(c)[0]
                            if tc != tokenizer.unk_token:
                                candidates.append(tc)
                        for c in random.sample(pairs[2], candidates_num - a):
                            tc = tokenizer.tokenize(c)[0]
                            if tc != tokenizer.unk_token:
                                candidates.append(tc)
                        confusion_set[pairs[0]] = candidates

            for key in confusion_set.keys():
                candidates = confusion_set[key]
                candidates = tokenizer.convert_tokens_to_ids(candidates)
                confusion_set[key] = candidates

            all_src_ids, all_attention_mask, all_segment_ids, all_tgt_ids = [], [], [], []
            all_candidates_ids = []
            for src, tgt in example_list:
                src_tokens = tokenizer.tokenize(src)
                tgt_tokens = tokenizer.tokenize(tgt)
                src_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]
                tgt_tokens = ["[CLS]"] + tgt_tokens + ["[SEP]"]

                src_ids = tokenizer.convert_tokens_to_ids(src_tokens)
                tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)

                candidates_ids = []
                for s in src_tokens:
                    single_candidates_ids = confusion_set[s]
                    if len(single_candidates_ids) > candidates_num:
                        single_candidates_ids = random.sample(single_candidates_ids, candidates_num)
                    single_candidates_ids = single_candidates_ids + [0] * (candidates_num - len(single_candidates_ids))
                    candidates_ids.append(single_candidates_ids)

                attention_mask = [1] * len(src_tokens)

                delta_length = max_len - len(src_tokens)
                src_ids = src_ids + [0] * delta_length
                tgt_ids = tgt_ids + [0] * delta_length
                attention_mask = attention_mask + [0] * delta_length
                segment_ids = [0] * max_len
                candidate_pad = [0] * candidates_num
                candidates_ids = candidates_ids + [candidate_pad] * delta_length

                all_src_ids.append(src_ids)
                all_attention_mask.append(attention_mask)
                all_segment_ids.append(segment_ids)
                all_tgt_ids.append(tgt_ids)
                all_candidates_ids.append(candidates_ids)

            self.all_src_ids = all_src_ids
            self.all_attention_mask = all_attention_mask
            self.all_segment_ids = all_segment_ids
            self.all_tgt_ids = all_tgt_ids
            self.all_candidates_ids = all_candidates_ids

            if use_cache:
                dataset = {
                    "src_ids": self.all_src_ids,
                    "attention_mask": self.all_attention_mask,
                    "segment_ids": self.all_segment_ids,
                    "tgt_ids": self.all_tgt_ids,
                    "candidates_ids": self.all_candidates_ids
                }
                torch.save(dataset, cache_file)

    def __len__(self):
        return len(self.all_src_ids)

    def __getitem__(self, index):
        return (
            torch.tensor(self.all_src_ids[index], dtype=torch.long),
            torch.tensor(self.all_attention_mask[index], dtype=torch.long),
            torch.tensor(self.all_segment_ids[index], dtype=torch.long),
            torch.tensor(self.all_candidates_ids[index], dtype=torch.long),
            torch.tensor(self.all_tgt_ids[index], dtype=torch.long),
        )

