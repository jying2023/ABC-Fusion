import numpy as np
import operator

def metric_file(sour_file, pred_file, preds, targs, sours, epoch):
   # preds, targs, sours=[],[],[]
    lines = open(sour_file, 'r', encoding='utf-8').readlines()
    f = open(pred_file + "output_{}.txt".format(epoch), 'w', encoding='utf-8')
    f1 = open(pred_file + "error.txt", 'w', encoding='utf-8')
    assert len(preds) == len(targs)==len(sours)
    for line, token in zip(lines,preds):
        line = line.strip()
        pairs = line.split(" ")
        pred = post_process(token, pairs[0])
        if len(pred) != len(pairs[1]):
            f1.write(str(token)+"\n"+pred+"\n"+pairs[0]+"\n"+pairs[1]+"\n")
        f.write(pred+"\n")
    f1.close()
    f.close()

    results13 = sent_metric(preds, targs, sours,'SIGHAN13')
    results14 = sent_metric(preds, targs, sours,'SIGHAN14')
    results15 = sent_metric(preds, targs, sours,'SIGHAN15')
    results_all = sent_metric(preds, targs, sours, 'ALL')

    results = {
        'SIGHAN13': results13,
        'SIGHAN14': results14,
        'SIGHAN15': results15,
        'ALL': results_all,
    }
    return results



data_interval = {
    'SIGHAN13': [0, 1000],
    'SIGHAN14': [1000, 1000+1062],
    'SIGHAN15': [1000+1062, 1000+1062+1100],
    'ALL': [0, 1000+1062+1100]
}

def sent_metric(preds, targs, sours, data_name):
    assert len(preds) == len(targs)
    m = data_interval[data_name][0]
    n = data_interval[data_name][1]
    sours, targs, preds = sours[m:n], targs[m:n], preds[m:n]
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    right_pred_all, check_right_pred_all = 0, 0
    total_num = len(preds)
    for sour, targ, pred in zip(sours, targs, preds):
        # assert len(sour) == len(targ)
        # assert len(targ) == len(pred)
        if len(sour) < len(pred):
            a = len(pred) - len(sour)
            pred = pred[0:-a]
        gold_errs = [idx for (idx, tk) in enumerate(targ) if tk != sour[idx]]
        pred_errs = [idx for (idx, tk) in enumerate(pred) if tk != sour[idx]]
        if len(gold_errs) > 0:
            total_gold_err += 1
        if len(pred_errs) > 0:
            total_pred_err += 1
            if gold_errs == pred_errs:
                check_right_pred_err += 1
            if targ == pred:
                right_pred_err += 1
        if gold_errs == pred_errs:
            check_right_pred_all += 1
        if targ == pred:
            right_pred_all += 1
    detect_precision = 1. * check_right_pred_err / total_pred_err
    detect_recall = 1. * check_right_pred_err / total_gold_err
    detect_f1 = 2 * detect_precision * detect_recall / (detect_precision + detect_recall + 1e-13)
    detect_accuracy = 1. * check_right_pred_all / total_num

    correct_precision = 1. * right_pred_err / total_pred_err
    correct_recall = 1. * right_pred_err / total_gold_err
    correct_f1 = 2 * correct_precision * correct_recall / (correct_precision + correct_recall + 1e-13)
    correct_accuracy = 1. * right_pred_all / total_num
    results = {
            'detect-acc': detect_accuracy,
            'detect-p': detect_precision,
            'detect-r': detect_recall,
            'detect-f1': detect_f1,
            'correct-acc': correct_accuracy,
            'correct-p': correct_precision,
            'correct-r': correct_recall,
            'correct-f1': correct_f1,
        }
    return results


def post_process(pred_tokens, source):
    pred = pred_tokens.copy()
    for i in range(len(pred)):
        if i > len(pred) -1 :
            break
        if pred[i] == '[UNK]':
            while i+1 < len(pred) - 1 and pred[i+1] == '[UNK]':
                del pred[i+1]
            if i == 0:
                m = 0
            else:
                s = "".join(pred[0:i])
                m = len(s)
                #m = source.find(s)+len(s)
            source1 = source[m:]
            if i+1 <= len(pred) - 1:
                ed = source1.find(pred[i+1])
            else:
                ed = len(source)-1
            pred[i] = source[m:ed+m]
    return "".join(pred)

