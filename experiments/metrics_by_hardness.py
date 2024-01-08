#%%
# from random import setstate
from collections import Counter
from metrics_util import get_hardness
from sklearn import metrics
import numpy as np
import json
from collections import namedtuple
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import os

SPIDER_HARDNESS = ['easy', 'medium', 'hard', 'extra']

Performance = namedtuple("Perf", "name acc pprec prec pf1 nprec nrec nf1 auc")
def get_optimal_threshold_for_metric(held_out_json, metric:str, flip_pos_label, verbose=False):
    assert metric in ['+f1', '-f1', 'acc']
    pos_label=1
    if flip_pos_label:
        # pos_rel = False
        pos_label=0
        for pred in held_out_json:
            pred['label'] = 1 - pred['label']
        
    sorted_pred = sorted(held_out_json, key=lambda x: x[PREDICTION_KEY])
    # Initialization
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    max_f11 = 0
    max_f11_tr = 0
    max_f12 = 0
    max_f12_tr = 0
    max_acc = 0
    max_acc_tr = 0
    correct = 0
    for pred in sorted_pred:
        if pred['label'] == pos_label:
            correct += 1
        if flip_pos_label:
            # threshold=0,  all predictions are negative
            if pred['label'] == pos_label:
                fn += 1
            else:
                tn += 1
        else:
            # threshold=0,  all predictions are positive
            if pred['label'] == pos_label:
                tp += 1
            else:
                fp += 1
    
    for sample in sorted_pred:
    # Get optimal threshold
        if flip_pos_label:
            # pred: neg -> pos
            if sample['label'] == pos_label:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
        else:
            # pred: pos -> neg
            if sample['label'] == pos_label:
                tp -= 1
                fn += 1
            else:
                fp -= 1
                tn += 1
        acc = (tp+tn)/(len(held_out_json))
        _threshold = sample[PREDICTION_KEY]

        f11 = 0
        f12 = 0
        if acc >= max_acc:
            max_acc = acc
            max_acc_tr = _threshold        
        if tp>0:
            prec1 = tp/(tp+fp)
            rec1 = tp/(tp+fn)
            f11 = 2*(prec1*rec1)/(prec1+rec1) # +F1
        if f11 >= max_f11:
            max_f11 = f11
            max_f11_tr = _threshold
        if tn > 0:
            prec2 = tn/(tn+fn)
            rec2 = tn/(tn+fp)
            f12 = 2*(prec2*rec2)/(prec2+rec2) # -F1
        if f12 >= max_f12:
            max_f12 = f12
            max_f12_tr = _threshold
    trs = {
        'acc': max_acc_tr,
        '+f1': max_f11_tr,
        '-f1': max_f12_tr
    }
    if verbose:
        print(f'Parser acc on held out set is {correct}/{len(held_out_json)} = {correct/len(held_out_json)}')
        print(f'Best threshold for {metric} is: {trs[metric]}')
    return trs[metric]
        
        
def get_held_out_and_test_samples(eval_json, held_out_ids, dataset):
    if '_beam' in dataset:
        dataset = dataset.replace('_beam', '')
    ref_file = f'../preprocessing/datasets/{dataset}/ed_{dataset}_beam_test.json'


    with open('../data/'+ref_file) as f:
        ref_json = json.load(f)
    held_out_eval_json = []
    test_eval_json = []

    for sample in eval_json:
        if ref_json[sample['id']][0]['idx'] not in held_out_ids:

            held_out_eval_json.append(sample)
        else:
            test_eval_json.append(sample)
    return held_out_eval_json, test_eval_json


#%%
def get_auc(eval_json, pos_label=0):
    score_list = []
    label_list = []
    for data in eval_json:
        label_list.append(data['label'])
        # score_list.append(1-data[PREDICTION_KEY])
        score_list.append(data[PREDICTION_KEY])
    labels = np.array(label_list)
    scores = np.array(score_list)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=pos_label)
    auc = np.round(metrics.auc(fpr,tpr),3)
    return fpr, tpr, auc
def analyze_experiment_k_fold(expr_name, n_fold, setting='top1', verbose=True, threshold=0.8, dump_errors=False, optimize_metric=None, ref_file=None):
    with open(f'../preprocessing/datasets/{DATASET_NAME}/ed_{DATASET_NAME}_beam_test_sim2.json') as f:
        ref_json = json.load(f)

    dataset_name = DATASET_NAME

    with open(f'../../data/spider/spider_dev_{n_fold}_fold.json') as f:
        k_fold_json = json.load(f)
    k_fold_perfs = []
    


    eval_filename = f'{expr_name}/eval_{dataset_name}_{setting}.json'
    with open(eval_filename) as f:
        eval_json = json.load(f)
    
    keep_top1 = True
    if 'beam' in expr_name or 'beam' in setting:
        if keep_top1:
            new_eval_json = [beam[0] for beam in eval_json]
        else:
            new_eval_json = []
            for beam in eval_json:
                new_eval_json.extend(beam)
    
        eval_json = new_eval_json

    
    if ref_json is not None:
        for idx, ref in enumerate(ref_json):
            eval_json[idx]['label'] = ref[0]['label']
            hardness = get_hardness(ref[0]['gold'], ref[0]['db_id'])
            eval_json[idx]['hardness'] = hardness
            

    for fold in k_fold_json:
        perf = analyze_experiment_by_hardness(expr_name, eval_json, ref_json=ref_json, held_out_ids = fold['held_out_ids'], setting=setting, verbose=verbose,threshold=threshold,dump_errors=dump_errors,optimize_metric=optimize_metric, ref_file=ref_file)
        k_fold_perfs.append(perf)
    
    k_fold_perfs_by_hardness = {}
    for hardness in SPIDER_HARDNESS:
        perfs = [k_fold_perf[hardness] for k_fold_perf in k_fold_perfs]
        k_fold_perfs_by_hardness[hardness] = mean_perf(perfs)
    return k_fold_perfs_by_hardness



def mean_perf(perf_list):
    return Performance(
        perf_list[0].name,
        np.mean([p.acc for p in perf_list]),
        np.mean([p.pprec for p in perf_list]),
        np.mean([p.prec for p in perf_list]),
        np.mean([p.pf1 for p in perf_list]),
        np.mean([p.nprec for p in perf_list]),
        np.mean([p.nrec for p in perf_list]),
        np.mean([p.nf1 for p in perf_list]),
        np.mean([p.auc for p in perf_list]),
    )
def get_performance(expr_name, eval_json, flip_pos_label=False, threshold=0.5, dump_errors=False, dataset_json=None):
    if verbose:
        print('-'*40)
        print('threshold: ', threshold)
        print()
        print(expr_name)
    
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    correct = 0
    fp_list = []
    fn_list = []
    pos_label = 1
    
    for idx, sample in enumerate(eval_json):
        # pred  = sample['pred']
        
        pred = 1 if sample[PREDICTION_KEY] > threshold else 0
        if flip_pos_label:
            pred = 1-pred
        
        gold = sample['label']

        if pred == gold:
            correct += 1
            if gold == pos_label:
                tp += 1
            else:
                tn += 1
        else:
            # if idx > len(dataset_json):
                # print(len(dataset_json))
                # print(idx)
            
            if gold == pos_label:
                fn += 1
                if dump_errors:
                    test_sample = dataset_json[idx]
                    err_sample= {
                        'idx': idx,
                        'label': test_sample['label'],
                        'question': test_sample['question'],
                        'sql': test_sample['sql'],
                        'gold': test_sample['gold'],
                        'ed_score': sample[PREDICTION_KEY],
                        'ed_threshold': threshold,
                        'parser_score': test_sample['confidence']
                    }
                    fn_list.append(err_sample)
            else:
                fp += 1
                if dump_errors:
                    err_sample= {
                        'idx': idx,
                        'label': test_sample['label'],
                        'question': test_sample['question'],
                        'sql': test_sample['sql'],
                        'gold': test_sample['gold'],
                        'ed_score': sample[PREDICTION_KEY],
                        'ed_threshold': threshold,
                        'parser_score': test_sample['confidence']
                    }
                    fp_list.append(err_sample)
    
    prec1 = 0
    rec1 = 0
    f11 = 0
    prec2 = 0
    rec2 = 0
    f12 = 0
    if tp+tn > 0:
        acc = (tp+tn)/(len(eval_json))
    if tp > 0:
        prec1 = tp/(tp+fp)
        rec1 = tp/(tp+fn)
        f11 = 2*(prec1*rec1)/(prec1+rec1)
    else:
        f11 = 0
    if tn > 0:
        prec2 = tn/(tn+fn)
        rec2 = tn/(tn+fp)
        f12 = 2*(prec2*rec2)/(prec2+rec2)
    else:
        f12 = 0
    if flip_pos_label: 
        pos_label = 1-pos_label
    _, _, auc = get_auc(eval_json, pos_label=pos_label)
    if verbose:
        print(f'errors: {tn+fp}/{tn+fp+fn+tp}')
        print('----')
        print('acc:  ',acc)
        print('++++')
        print(f'tp:{tp} tn:{tn} fn: {fn} fp: {fp}')
        print('prec: ', prec1)
        print('rec : ', rec1)
        print('f1:   ', f11)
        print('----')
        print(f'tp:{tp} tn:{tn} fn: {fn} fp: {fp}')
        print('prec: ', prec2)
        print('rec : ', rec2)
        print('f1:   ', f12)
        print()
        print('auc: ', auc)
        print('overall f1: ', (f11+f12)/2)    
    
    if dump_errors:
        expr_name1 = expr_name.split('/')[-1]
        with open(f'Errors/{expr_name1}_fp.json', 'w') as f:
            json.dump(fp_list, f, indent=2)
        with open(f'Errors/{expr_name1}_fn.json', 'w') as f:
            json.dump(fn_list, f, indent=2)

    return Performance(
        expr_name, 
        acc, 
        pf1 = f11, 
        pprec=prec1, 
        prec=rec1,
        nf1=f12,
        nprec=prec2,
        nrec=rec2,
        auc=auc
        )
def analyze_experiment_by_hardness(expr_name, eval_json, ref_json=None, held_out_ids = [], setting='top1', verbose=False, threshold=0.8, dump_errors=False, optimize_metric=None, ref_file=None):
    
    flip_pos_label = 'mc' in expr_name
    if optimize_metric is not None:
        held_out, eval_json = get_held_out_and_test_samples(eval_json, held_out_ids, DATASET_NAME)
        # eval_json=held_out
    if optimize_metric is not None:
        threshold = get_optimal_threshold_for_metric(held_out, metric=optimize_metric, flip_pos_label=flip_pos_label)
    samples_by_hardness = {
        'easy': [],
        'medium': [],
        'hard': [],
        'extra': [],
    }
    for eval in eval_json:
        samples_by_hardness[eval['hardness']].append(eval)
    return {
        'easy': get_performance(expr_name, samples_by_hardness['easy'], threshold=threshold, flip_pos_label=flip_pos_label, dump_errors=False),
        'medium': get_performance(expr_name, samples_by_hardness['medium'], threshold=threshold, flip_pos_label=flip_pos_label, dump_errors=False),
        'hard': get_performance(expr_name, samples_by_hardness['hard'], threshold=threshold, flip_pos_label=flip_pos_label, dump_errors=False),
        'extra': get_performance(expr_name, samples_by_hardness['extra'], threshold=threshold, flip_pos_label=flip_pos_label, dump_errors=False),
    }
    

    

def mean_and_var(exprs:list, setting='top1', optimize_metric=None,verbose=False, ref_file=None):
    print('=' * 20)
    print(exprs[0])
    acc_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    pf1_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    pprec_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    prec_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    nf1_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    nrec_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    nprec_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    auc_list_by_hardness = {'easy':[], 'medium':[], 'hard':[], 'extra':[]}
    
    
    for expr in exprs:
        tr = 0.5
        if 'mc' in expr:
            tr = 0.1
        if 'mc' and 'logit' in expr:
            tr = 0.5
        elif 'mc' and 'score' in expr:
            tr = 0.2
        # perf = analyze_experiment(expr, setting=setting, verbose=False, threshold=tr)
        perf_by_hardness = analyze_experiment_k_fold(expr,5,setting=setting, threshold=tr, optimize_metric=optimize_metric,verbose=verbose, ref_file=ref_file)
        
        for hardness in SPIDER_HARDNESS:
            perf = perf_by_hardness[hardness]
            acc_list_by_hardness[hardness].append(perf.acc)
            pf1_list_by_hardness[hardness].append(perf.pf1)
            pprec_list_by_hardness[hardness].append(perf.pprec)
            prec_list_by_hardness[hardness].append(perf.prec)
            nf1_list_by_hardness[hardness].append(perf.nf1)
            nrec_list_by_hardness[hardness].append(perf.nrec)
            nprec_list_by_hardness[hardness].append(perf.nprec)
            auc_list_by_hardness[hardness].append(perf.auc)
    rounding_digit = 1
    for hardness in SPIDER_HARDNESS:
        pprec_list = pprec_list_by_hardness[hardness]
        print('='*10, hardness, '='*10)
        pprec_list = pprec_list_by_hardness[hardness]
        print(f'+prec: {np.round(np.mean(pprec_list)*100,rounding_digit)}({np.round(np.std(pprec_list)*100,rounding_digit)})')
        prec_list = prec_list_by_hardness[hardness]
        print(f'+rec: {np.round(np.mean(prec_list)*100,rounding_digit)}({np.round(np.std(prec_list)*100,rounding_digit)})')
        pf1_list = pf1_list_by_hardness[hardness]
        print(f'+f1: {np.round(np.mean(pf1_list)*100,rounding_digit)}({np.round(np.std(pf1_list)*100,rounding_digit)})')
        print('-'*10)
        nprec_list = nprec_list_by_hardness[hardness]
        print(f'-prec: {np.round(np.mean(nprec_list)*100,rounding_digit)}({np.round(np.std(nprec_list)*100,rounding_digit)})')
        nrec_list = nrec_list_by_hardness[hardness]
        print(f'-rec: {np.round(np.mean(nrec_list)*100,rounding_digit)}({np.round(np.std(nrec_list)*100,rounding_digit)})')
        nf1_list = nf1_list_by_hardness[hardness]
        print(f'-f1: {np.round(np.mean(nf1_list)*100,rounding_digit)}({np.round(np.std(nf1_list)*100,rounding_digit)})')
        print('-'*10)
        acc_list = acc_list_by_hardness[hardness]
        print(f'acc: {np.round(np.mean(acc_list)*100,rounding_digit)}({np.round(np.std(acc_list)*100,rounding_digit)})')
        auc_list = auc_list_by_hardness[hardness]
        print(f'auc: {np.round(np.mean(auc_list)*100,rounding_digit)}({np.round(np.std(auc_list)*100,rounding_digit)})')
        


def get_expr_series(expr_name, num_exps):
    expr_list = []
    for i in range(1, num_exps + 1):
        expr_list.append(f'{expr_name}_{i}')
    return expr_list


#%%
# DATASET_NAME = 'bridge'
# DATASET_NAME= 'natsql'
# DATASET_NAME = 'smbop'
DATASET_NAME = 'resdnatsql'

PREDICTION_KEY= 'score'



#%%
if __name__ == '__main__':

    exprs = [
        
        ('NatSQL/CodeBERT', 1, 'NatSQL'),
        ('NatSQL/CodeBERT_GAT', 1, 'NatSQL'),
        ('ResdNatSQL/CodeBERT', 1, 'ResdNatSQL'),
        ('ResdNatSQL/CodeBERT_GAT', 1, 'ResdNatSQL'),
        ('SmBoP/CodeBERT', 1, 'SmBoP'),
        ('SmBoP/CodeBERT_GAT', 1, 'SmBoP'),
    ]
    
    verbose = False
    setting = 'beam'
    optimize_metric = 'acc'
    if 'dev' in DATASET_NAME:
        optimize_metric = None
    ref_file=None
    for expr in exprs:
        mean_and_var(get_expr_series(expr[0], expr[1]), setting=setting, optimize_metric=optimize_metric, verbose=verbose, ref_file=ref_file)
    exit()
