import os
import sys
import argparse
import torch
import math
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import set_seed, BertTokenizer, BertConfig, AdamW
from cscbert import CSCBert
from cscdataset import ConfuseDataSet
from train_confusion import Trainer
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='the directory of the dataset.')
    parser.add_argument('--train_file', type=str, default='train.txt', help='the filename of the training data.')
    parser.add_argument('--valid_file', type=str, default='test.txt', help='the filename of the validation data.')
    parser.add_argument('--test_file', type=str, default='test.txt', help='the filename of the test data.')
    parser.add_argument('--model_name_or_path', type=str, default=None, help='the path or name of pretrained model.')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for the training dataloader.')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='batch size for the validation dataloader.')
    parser.add_argument('--test_batch_size', type=int, default=32, help='batch size for the test dataloader.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='initial learning rate.')
    parser.add_argument('--adapter_learning_rate', type=float, default=2e-4, help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay to use.')
    parser.add_argument('--num_train_epochs', type=int, default=8, help='total number of training epochs.')
    parser.add_argument('--inject_position', nargs='+', type=int, default=[1], help='total number of training epochs.')
    parser.add_argument('--candidates_num', type=int, default=5, help='total number of training epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps.')
    parser.add_argument('--warmup_portion', type=float, default=0, help='number of warmup steps.')
    parser.add_argument('--output_dir', type=str, default=None, help='the directory to store the model and output.')
    parser.add_argument('--max_seq_length', type=int, default=200, help='the maximum input sequence length.')
    parser.add_argument('--seed', type=int, default=666, help='a seed for reproducible training.')
    parser.add_argument('--load_model', action='store_true', default=False, help='whether to load model from checkpoints.')


    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    set_seed(args.seed)

    handler = logging.FileHandler(args.output_dir + "log.txt",mode='a+')
    logger.addHandler(handler)

    config = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = CSCBert.from_pretrained(args.model_name_or_path)
    if args.load_model and os.path.exists(args.output_dir+'ALL_model.pt') :
        model.load_state_dict(torch.load(args.output_dir+'ALL_model.pt'))

    model.cuda()
    train_dataset = ConfuseDataSet(args.data_dir + args.train_file, tokenizer=tokenizer, max_len=args.max_seq_length, use_cache=True, candidates_num=args.candidates_num, subset='train')
    eval_dataset = ConfuseDataSet(args.data_dir + args.valid_file, tokenizer=tokenizer, max_len=args.max_seq_length, use_cache=True, candidates_num=args.candidates_num, subset='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.valid_batch_size, shuffle=False)
    args.max_train_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_warmup_steps = math.ceil(args.max_train_steps * args.warmup_portion)

    logger.info(args)
    #no_decay = []
    no_decay = ["bias", "LayerNorm.weight"]
    special_lr = ["bias", "LayerNorm.weight","adapter"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in special_lr)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if ("adapter" in n) and (not any(nd in n for nd in no_decay))],
            "weight_decay": args.weight_decay,
            "lr": args.adapter_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if ("adapter" in n) and (any(nd in n for nd in no_decay))],
            "weight_decay": 0.0,
            "lr": args.adapter_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if ("adapter" not in n) and (any(nd in n for nd in no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    trainer = Trainer(args, tokenizer, model, optimizer, logger)
    trainer.train(train_dataloader, eval_dataloader)

if __name__ == '__main__':
    main()
