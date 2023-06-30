import copy
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import BertModel, BertConfig, BertTokenizer, get_scheduler
from metric import metric_file, sent_metric
import pdb

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer():
    def __init__(self, args, tokenizer, model, optimizer, logger):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = get_scheduler(
            name='constant',
            optimizer=self.optimizer,
        )
        self.scaler = GradScaler()
        self.logger = logger

    def train(self, train_dataloader, eval_dataloader, subset='ALL'):
        best_loss, best_acc, best_f1, best_epoch = 0., 0., 0., 0
        num_epoch = self.args.num_train_epochs
        if subset != 'ALL':
            num_epoch = self.args.num_single_train_epochs
        for epoch in range(num_epoch):
            train_loss, train_acc = self.train_singel_epoch(train_dataloader)
            eval_loss, eval_acc, outputs = self.evaluate(eval_dataloader)
            self.logger.info("epoch {:d}, train_loss: {:.4f}, eval_loss: {:.4f}".format(epoch, train_loss, eval_loss))
            self.logger.info("epoch {:d}, train_acc: {:.4f}, eval_acc: {:.4f}".format(epoch, train_acc, eval_acc))

            save_file = self.args.output_dir +subset+ '_model.pt'
            torch.save(self.model.state_dict(), save_file)

            f1_score = self.compute_metric(*outputs,epoch,subset)

            if f1_score > best_f1:
                save_file = self.args.output_dir +subset+ '_best_model.pt'
                torch.save(self.model.state_dict(), save_file)
                best_acc = eval_acc
                best_loss = eval_loss
                best_f1 = f1_score
                best_epoch = epoch
                self.logger.info("best_epoch {:d}, best_loss: {:.4f}, best_acc: {:.4f}, best_f1: {:.4f}".format(best_epoch, best_loss, best_acc, best_f1))
        self.logger.info("Training finished.")
        self.logger.info("best_epoch {:d}, best_loss: {:.4f}, best_acc: {:.4f}, best_f1: {:.4f}".format(best_epoch, best_loss, best_acc, best_f1))

    def train_singel_epoch(self, train_dataloader):
        self.model.train()

        losses = AverageMeter()
        accs = AverageMeter()

        self.optimizer.zero_grad()
        pbar = tqdm(train_dataloader, total=len(train_dataloader), position=0, leave=True)
        for step, batch in enumerate(pbar):
            batch = [t.cuda() for t in batch]
            with autocast():
                loss, output = self.model(*batch, inject_position=self.args.inject_position) 
                loss = loss / self.args.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

            if step % self.args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()  

            label = batch[-1].view(-1)
            non_pad_mask = label.ne(0)
            ntokens = non_pad_mask.sum().item()
            acc = (output.argmax(-1).view(-1)[non_pad_mask] == label[non_pad_mask]).sum() / ntokens 
            losses.update(loss.item() * self.args.gradient_accumulation_steps, ntokens)
            accs.update(acc.item(), ntokens)
            pbar.set_postfix(loss=losses.avg, acc=accs.avg)
        return losses.avg, accs.avg

    def evaluate(self, eval_dataloader):
        self.model.eval()
        losses = AverageMeter()
        accs = AverageMeter()
        pred_tokens = []
        sour_tokens = []
        targ_tokens = []

        pbar = tqdm(eval_dataloader, total=len(eval_dataloader), position=0, leave=True)
        for step, batch in enumerate(pbar):
            batch = [t.cuda() for t in batch]
            with torch.no_grad():
                loss, output = self.model(*batch)
            bsz, seqlen = batch[0].size()
            label = batch[-1].view(-1)
            non_pad_mask = label.ne(0)  
            ntokens = non_pad_mask.sum().item()
            acc = (output.argmax(-1).view(-1)[non_pad_mask] == label[non_pad_mask]).sum() / ntokens 
            losses.update(loss.item() * self.args.gradient_accumulation_steps, ntokens)
            accs.update(acc.item(), ntokens)
            pbar.set_postfix(loss=losses.avg, acc=accs.avg)
            preds = output.argmax(-1) * non_pad_mask.view(bsz, seqlen)
            preds = preds.cpu().numpy()

            for i in range(bsz):
                p = self.decode(preds[i])
                pred_tokens.append(p)
                s = self.decode(batch[0][i])
                sour_tokens.append(s)
                d = self.decode(batch[-1][i])
                targ_tokens.append(d)

        return losses.avg, accs.avg, (pred_tokens, targ_tokens, sour_tokens)

    def compute_metric(self, preds, targs, sours,epoch,subset='ALL'):
        results = metric_file(self.args.data_dir + self.args.valid_file, self.args.output_dir, preds, targs, sours, epoch)
        self.logger.info('SIGHAN13: Detection-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['SIGHAN13']['detect-acc'], results['SIGHAN13']['detect-p'], results['SIGHAN13']['detect-r'], results['SIGHAN13']['detect-f1']))
        self.logger.info('SIGHAN13: Correction-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['SIGHAN13']['correct-acc'], results['SIGHAN13']['correct-p'], results['SIGHAN13']['correct-r'], results['SIGHAN13']['correct-f1']))
        self.logger.info('SIGHAN14: Detection-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['SIGHAN14']['detect-acc'], results['SIGHAN14']['detect-p'], results['SIGHAN14']['detect-r'], results['SIGHAN14']['detect-f1']))
        self.logger.info('SIGHAN14: Correction-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['SIGHAN14']['correct-acc'], results['SIGHAN14']['correct-p'], results['SIGHAN14']['correct-r'], results['SIGHAN14']['correct-f1']))
        self.logger.info('SIGHAN15: Detection-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['SIGHAN15']['detect-acc'], results['SIGHAN15']['detect-p'], results['SIGHAN15']['detect-r'], results['SIGHAN15']['detect-f1']))
        self.logger.info('SIGHAN15: Correction-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['SIGHAN15']['correct-acc'], results['SIGHAN15']['correct-p'], results['SIGHAN15']['correct-r'], results['SIGHAN15']['correct-f1']))
        self.logger.info('ALL:      Detection-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['ALL']['detect-acc'], results['ALL']['detect-p'], results['ALL']['detect-r'], results['ALL']['detect-f1']))
        self.logger.info('ALL:      Correction-level: acc:{:.4f}, p:{:.4f}, r:{:.4f}, f1:{:.4f}'.format(results['ALL']['correct-acc'], results['ALL']['correct-p'], results['ALL']['correct-r'], results['ALL']['correct-f1']))
        return results[subset]['correct-f1']

    def test(self, test_dataloader):
        self.model.eval()
        pbar = tqdm(test_dataloader, total=len(test_dataloader), position=0, leave=True)
        f = open(self.args.output_dir + 'output.txt', 'w', encoding='utf-8')
        for step, batch in enumerate(pbar):
            batch = [t.cuda() for t in batch]
            with torch.no_grad():
                _, output = self.model(*batch)

            non_pad_mask = batch[-1].ne(0)
            preds = output.argmax(-1) * non_pad_mask
            preds = preds.cpu().numpy()
            for p in preds:
                tokens = self.decode(p)
                f.write(tokens + '\n')
        f.close()
        results = metric_file(self.args.data_dir + self.args.test_file, self.args.output_dir + 'output.txt')
        for k, v in results.items():
            print(f'{k}: {v}, ', end='')
        print('')

    def decode(self, token_ids):
        tokens =self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        if '[CLS]' in tokens:
            cls_index = tokens.index('[CLS]')
            del tokens[cls_index]
        if '[SEP]' in tokens:
            sep_index = tokens.index('[SEP]')
            tokens = tokens[0:sep_index]
        if '[PAD]' in tokens:
            pad_index = tokens.index('[PAD]')
            tokens = tokens[0:pad_index]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].replace("##","")
        return tokens
