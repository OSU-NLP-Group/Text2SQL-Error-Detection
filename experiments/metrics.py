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
    
    with open(ref_file) as f:
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
        score_list.append(data[PREDICTION_KEY])
    labels = np.array(label_list)
    scores = np.array(score_list)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=pos_label)
    auc = np.round(metrics.auc(fpr,tpr),3)
    return fpr, tpr, auc
def analyze_experiment_k_fold(expr_name, n_fold, setting='top1', verbose=True, threshold=0.8, dump_errors=False, optimize_metric=None, ref_file=None):
    dataset_name = DATASET_NAME
    with open(f'../../data/spider/spider_dev_{n_fold}_fold.json') as f:
    
        k_fold_json = json.load(f)
    k_fold_perfs = []
    for fold in k_fold_json:
        perf = analyze_experiment(expr_name, held_out_ids = fold['held_out_ids'], setting=setting, verbose=verbose,threshold=threshold,dump_errors=dump_errors,optimize_metric=optimize_metric, ref_file=ref_file)
        k_fold_perfs.append(perf)
        
    return mean_perf(k_fold_perfs)
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
def analyze_experiment(expr_name, held_out_ids = [], setting='top1', verbose=False, threshold=0.8, dump_errors=False, optimize_metric=None, ref_file=None):
    # dataset_name = 'lgesql'
    dataset_name = DATASET_NAME
    if dump_errors:
        with open(f'../data/ed_{dataset_name}_test_top1_gg.json') as f:
            dataset_json = json.load(f)
        
    ref_json = None
    
    eval_filename = f'{expr_name}/eval_{dataset_name}_{setting}.json'
    
    with open(eval_filename) as f:
        eval_json = json.load(f)
    keep_top1 = True
    if 'beam' in expr_name or 'beam' in setting:
        # assert ref_file == None
        if keep_top1:
            new_eval_json = [beam[0] for beam in eval_json]
        else:
            new_eval_json = []
            for beam in eval_json:
                new_eval_json.extend(beam)
    
        eval_json = new_eval_json

    if ref_file is not None:
        with open(ref_file) as f:
            ref_json = json.load(f)
        for idx, ref in enumerate(ref_json):
            eval_json[idx]['label'] = ref[0]['label']

    
    flip_pos_label = 'mc' in expr_name
    if optimize_metric is not None:
        held_out, eval_json = get_held_out_and_test_samples(eval_json, held_out_ids, DATASET_NAME)
        # eval_json=held_out
    if optimize_metric is not None:
        threshold = get_optimal_threshold_for_metric(held_out, metric=optimize_metric, flip_pos_label=flip_pos_label)
    

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
    # if tp+tn > 0:
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

def mean_and_var(exprs:list, setting='top1', optimize_metric=None,verbose=False, ref_file=None):
    print('=' * 20)
    print(exprs[0])
    acc_list = []
    pf1_list = []
    pprec_list = []
    prec_list = []
    nf1_list = []
    nrec_list = []
    nprec_list = []
    auc_list = []
    
    
    for expr in exprs:
        tr = 0.5
        if 'mc' in expr:
            tr = 0.1
        if 'mc' and 'logit' in expr:
            tr = 0.5
        elif 'mc' and 'score' in expr:
            tr = 0.2
        # perf = analyze_experiment(expr, setting=setting, verbose=False, threshold=tr)
        perf = analyze_experiment_k_fold(expr,5,setting=setting, threshold=tr, optimize_metric=optimize_metric,verbose=verbose, ref_file=ref_file)
        acc_list.append(perf.acc)
        pf1_list.append(perf.pf1)
        pprec_list.append(perf.pprec)
        prec_list.append(perf.prec)
        nf1_list.append(perf.nf1)
        nrec_list.append(perf.nrec)
        nprec_list.append(perf.nprec)
        auc_list.append(perf.auc)
    rounding_digit = 1
    # print(auc_list)
    print(f'+prec: {np.round(np.mean(pprec_list)*100,rounding_digit)}({np.round(np.std(pprec_list)*100,rounding_digit)})')
    print(f'+rec: {np.round(np.mean(prec_list)*100,rounding_digit)}({np.round(np.std(prec_list)*100,rounding_digit)})')
    print(f'+f1: {np.round(np.mean(pf1_list)*100,rounding_digit)}({np.round(np.std(pf1_list)*100,rounding_digit)})')
    print('-'*10)
    print(f'-prec: {np.round(np.mean(nprec_list)*100,rounding_digit)}({np.round(np.std(nprec_list)*100,rounding_digit)})')
    print(f'-rec: {np.round(np.mean(nrec_list)*100,rounding_digit)}({np.round(np.std(nrec_list)*100,rounding_digit)})')
    print(f'-f1: {np.round(np.mean(nf1_list)*100,rounding_digit)}({np.round(np.std(nf1_list)*100,rounding_digit)})')
    print('-'*10)
    print(f'acc: {np.round(np.mean(acc_list)*100,rounding_digit)}({np.round(np.std(acc_list)*100,rounding_digit)})')
    print(f'auc: {np.round(np.mean(auc_list)*100,rounding_digit)}({np.round(np.std(auc_list)*100,rounding_digit)})')
    


def get_expr_series(expr_name, num_exps):
    expr_list = []
    for i in range(1, num_exps + 1):
        expr_list.append(f'{expr_name}_{i}')
    return expr_list


#%%
def get_data_by_varying_threshold(expr_name, plot_setting, dataset_name=None):
    # pos_rel = True
    if dataset_name == None:
        dataset_name = DATASET_NAME

    setting='beam'
    parser_name = dataset_name.lower().replace(' v2', '') # remove ' v2' from 'bridge v2'
    if os.path.exists(f'{expr_name}/eval_{parser_name}_{setting}_ot.json'):
        eval_filename = f'{expr_name}/eval_{parser_name}_{setting}_ot.json'
    else:
        eval_filename = f'{expr_name}/eval_{parser_name}_{setting}.json'
    
    with open(eval_filename) as f:
        eval_json = json.load(f)

    if 'beam' in setting:
        eval_json = [b[0] for b in eval_json]
    pos_label=1
    flip_pos_label = False
    if 'mc' in expr_name:
        # pos_rel = False
        flip_pos_label = True
        pos_label=0
        for pred in eval_json:
            pred['label'] = 1 - pred['label']

    sorted_pred = sorted(eval_json, key=lambda x: x[PREDICTION_KEY])
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    

    # Initialization
    for pred in sorted_pred:
        
        # if 'mc' in expr_name or PREDICTION_KEY == 'ot_sum_cost':
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


    model_name = '_'.join(expr_name.split('/')[-1].split('_')[:-1])
    if tp + fp > 0:
        precisions = [tp/(tp+fp)]
        N_pos_preds = [tp+fp]
        N_inter = [tn+fn]
        acc = [(tp+tn+fn)/len(eval_json)]
        if plot_setting == 'fig_precision-Q_answered':
            x_0 = [N_pos_preds[0]]
            y_0 = [precisions[0]]
        elif plot_setting == 'fig_acc-N_inter':
            x_0 = [N_inter[0]]
            y_0 = [acc[0]]
        elif plot_setting == 'fig_ROC':
            x_0 = [fp/(tn+fp)]
            y_0 = [tp/(tp+fn)]


        raw_df_dict = {
        'expr': [expr_name],
        'model': [model_name],
        'x': x_0,
        'y': y_0,
        'setting': [plot_setting],
        'dataset': [dataset_name]
        }
    else:
        precisions = []
        N_pos_preds = []
        N_inter = [tn+fn]
        acc = [(tp+tn+fn)/len(eval_json)]
        expr_0 = []
        model_0 = []
        setting_0 = []
        dataset_0 = []
        if plot_setting == 'fig_precision-Q_answered':
            x_0 = []
            y_0 = []
            dataset_0 = []
            dataset_0 = []
        elif plot_setting == 'fig_acc-N_inter':
            expr_0 = [expr_name]
            model_0 = [model_name]
            x_0 = [N_inter[0]]
            y_0 = [acc[0]]
            dataset_0 = [dataset_name]
            setting_0=[plot_setting]
        elif plot_setting == 'fig_ROC':
            expr_0 = [expr_name]
            model_0 = [model_name]
            x_0 = [fp/(tn+fp)]
            y_0 = [tp/(tp+fn)]
            setting_0=[plot_setting]
            dataset_0 = [dataset_name]


        raw_df_dict = {
        'expr': expr_0,
        'model': model_0, 
        'x': x_0,
        'y': y_0,
        'setting': setting_0,
        'dataset': dataset_0
        }

    for sample in sorted_pred:
        
        # if 'mc' in expr_name:
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

        if plot_setting=='fig_precision-Q_answered' and tp + fp > 0:    
            raw_df_dict['x'].append(tp+fp)
            raw_df_dict['y'].append(tp/(tp+fp))
            raw_df_dict['setting'].append(plot_setting)
            raw_df_dict['dataset'].append(dataset_name)
            raw_df_dict['expr'].append(expr_name)
            raw_df_dict['model'].append(model_name)
        elif plot_setting=='fig_acc-N_inter':
            raw_df_dict['dataset'].append(dataset_name)
            raw_df_dict['expr'].append(expr_name)
            raw_df_dict['model'].append(model_name)
            raw_df_dict['x'].append(tn+fn)
            raw_df_dict['y'].append((tp+tn+fn)/len(eval_json))
            raw_df_dict['setting'].append(plot_setting)
        elif plot_setting=='fig_ROC':
            raw_df_dict['dataset'].append(dataset_name)
            raw_df_dict['expr'].append(expr_name)
            raw_df_dict['model'].append(model_name)
            raw_df_dict['x'].append(fp/(tn+fp))
            raw_df_dict['y'].append(tp/(tp+fn))
            raw_df_dict['setting'].append(plot_setting)
    # return fpr, tpr
    return raw_df_dict


def get_line_plot_data(exprs, df, plot_setting, dataset_name):
    for expr in exprs:
        raw_df_dict = get_data_by_varying_threshold(expr, plot_setting, dataset_name)
        new_df = pandas.DataFrame(raw_df_dict)
        
        df1 = new_df.sort_values(['setting', 'model', 'x'], ascending=[True, True, 'acc' in plot_setting])
        df1 = df1[['model', 'x', 'y','setting']]
        df1 =df1[(df1['y']>=0.95)]
        # Print results for getting simulated interaction experiment metrics.
        print(df1.head(10))

        if df is None:
            df = new_df
        else:
            df = pandas.concat([df, new_df], ignore_index=True)
    print(df.shape)
    return df


# The script will look for the output eval_{DATASET_NAME}.json file.

# Main Experiments
# DATASET_NAME= 'natsql'
DATASET_NAME = 'smbop'
# DATASET_NAME = 'resdnatsql'

# Extra Experiments
# DATASET_NAME = 'bridge_kaggle'
# DATASET_NAME = 'bridge'

PREDICTION_KEY= 'score'


#%%
if __name__ == '__main__':
    print(f'Evaluating on the {DATASET_NAME} dataset')
    exprs = [
        # # Parser dependent baselines
        ('Parser_SmBop/SmBop_Prob', 1, 'SmBop'),
        ('Parser_SmBop/SmBop_mc_logit', 1, 'SmBop'),
        ('Parser_ResdNatSQL/RsedNatSQL_mc', 1, 'ResdNatSQL'),
        ('Parser_ResdNatSQL/ResdNatSQL_prob', 1, 'ResdNatSQL'),
        ('Parser_NatSQL/NatSQL_prob', 1,'NatSQL'),
        ('Parser_NatSQL/NatSQL_mc', 1,'NatSQL'),
        ('Parser_Bridge_v2/Bridge_v2_prob', 1, 'Bridge v2'),
        ('Parser_Bridge_v2/Bridge_v2_mc', 1, 'Bridge v2'),
        
        # # Main experiments
        ('NatSQL/CodeBERT', 1, 'NatSQL'),
        ('NatSQL/CodeBERT_GAT', 1, 'NatSQL'),
        ('ResdNatSQL/CodeBERT', 1, 'ResdNatSQL'),
        ('ResdNatSQL/CodeBERT_GAT', 1, 'ResdNatSQL'),
        ('SmBoP/CodeBERT', 1, 'SmBoP'),
        ('SmBoP/CodeBERT_GAT', 1, 'SmBoP'),

        # # Ablation
        # ('_NatSQL_full_beam_spider_tr/CodeBERT_b16_t-top1_e20',3,'NatSQL'),
        # ('_NatSQL_full_beam_ori/Proposed_mean_seq_cb_b16_lr3e-5_t-top1_e20', 3)
    ]
    

    # ######## Metrics
    verbose = False
    setting = 'beam'
    # Search for the best threshold in terms of accuracy during 5-fold cross-validation. 
    optimize_metric = 'acc'
    if 'dev' in DATASET_NAME:
        optimize_metric = None
    
    ref_file=None
    print(f'Evaluating performance on {DATASET_NAME}')
    print(f'Thresholds chosen to optimize {optimize_metric}.')
    for expr in exprs:
        mean_and_var(get_expr_series(expr[0], expr[1]), setting=setting, optimize_metric=optimize_metric, verbose=verbose, ref_file=ref_file)
    ########## Metrics


# ##### Plotting #####
# Uncomment the following code to plot figures for simulated interactive evaluations.

#     # For plotting simulated interactive experiment results, we only use the first model checkpoint.

    # exprs = [
    #     ('SmBop/SmBop_Prob', 1, 'SmBop'),
    #     ('SmBop/SmBop_mc_logit', 1, 'SmBop'),
    #     ('NatSQL/NatSQL_prob', 1,'NatSQL'),
    #     ('NatSQL/NatSQL_mc', 1,'NatSQL'),
    #     ('RESDNATSQL/RESDNATSQL_prob', 1, 'RESDNATSQL'),
    #     ('RESDNATSQL/RESDNATSQL_mc', 1, 'RESDNATSQL'),
        
    #     ('NatSQL/CodeBERT', 1, 'NatSQL'),
    #     ('NatSQL/CodeBERT_GAT', 1, 'NatSQL'),
    #     ('RESDNATSQL/CodeBERT', 1, 'RESDNATSQL'),
    #     ('RESDNATSQL/CodeBERT_GAT', 1, 'RESDNATSQL'),
    #     ('SmBoP/CodeBERT', 1, 'SmBoP'),
    #     ('SmBoP/CodeBERT_GAT', 1, 'SmBoP'),
    # ]

#     settings = ['fig_precision-Q_answered', 'fig_acc-N_inter' ]

#     df = None
#     for expr in exprs:
#         for plot_setting in settings:
#             df = get_line_plot_data(get_expr_series(expr[0], expr[1]), df=df, plot_setting=plot_setting, dataset_name=expr[2])
        


#     axes_labels = {
#         'fig_precision-Q_answered':{
#             'x': 'Questions Answered',
#             'y': 'Answer Triggering\nPrecision'
#         },
#         'fig_acc-N_inter':{
#             'x': 'Number of Interactions',
#             'y': 'Interaction Triggering\nAccuracy'
#         },
#         'fig_ROC': {
#             'x': 'False Positive Rate',
#             'y': 'True Positive Rate'
#         }
#     }
#     axes_titles = {
#         'fig_precision-Q_answered':'(a)Answer Triggering',
#         'fig_acc-N_inter':' (b) Interaction Triggering',
#     }

# #%%

#     # Create and save plots
#     FONT_SIZE = 30
#     plt.rc('font', size=FONT_SIZE)

#     fig, axes = plt.subplots(2,3, figsize=(30,15))
#     df_set = df.set_index('setting')
#     lines = []

#     for i, plot_setting in enumerate(settings):
#         for j, dataset_name in enumerate(['SmBop', 'RESDNATSQL', 'NatSQL']):
#             data = df_set.loc[plot_setting]
#             sns.lineplot(
#                 ax=axes[i][j],
#                 data=data[data['dataset']==dataset_name],
#                 x = 'x',
#                 y = 'y',
#                 hue='model',
#                 )
#             title=dataset_name
#             if dataset_name == 'Bridge v2':
#                 title = 'BRIDGE v2'
#             elif dataset_name == 'SmBop':
#                 title = 'SmBoP'
#             elif dataset_name == 'RESDNATSQL':
#                 title = 'RESDSQL'
#             if i == 0:
#                 axes[i][j].set_title(title, fontsize=FONT_SIZE,y=1.01)
            
#             axes[i][j].set_xlabel(axes_labels[plot_setting]['x'])
#             if j == 0:    
#                 axes[i][j].set_ylabel(axes_labels[plot_setting]['y'])
#             else:
#                 axes[i][j].set_ylabel('')
            
#             if i == 0:
#                 axes[i][j].set_ylim(0.65, 1.01)
#             elif i == 1: 
#                 axes[i][j].set_ylim(0.65, 1.01)
#             axes[i][j].get_legend().remove()

#     legend=plt.legend()
#     lines, labels = fig.axes[0].get_legend_handles_labels()
#     labels = [r'Probability', r'Dropout', r'CodeBert', r'CodeBert+GAT']

#     fig.axes[-1].get_legend().remove()
#     fig.legend(lines, labels, loc='upper center',ncol=5,fontsize=FONT_SIZE, bbox_to_anchor=(0.5,0.94))
#     plt.subplots_adjust(left=0.1,
#                         bottom=0.05,
#                         right=0.95,
#                         top=0.83,
#                         wspace=0.2,
#                         hspace=0.2)
#     # fig.savefig(f'output.pdf',bbox_inches='tight')
# ##### Plotting #####