import json
import numpy as np
def get_std(all_preds, idx,model='smbop'):    
    list = []
    for i in range(len(all_preds)):
        # try:
        if len(all_preds[i][idx]) == 0:
            continue
        
        if model in ['bridge']:
            list.append(all_preds[i][idx][1])
        elif model=='resdnatsql':
            list.append(all_preds[i][idx][0][1])
        elif model=='smbop':
            if all_preds[i][idx][-1] == '':
                continue
            list.append(all_preds[i][idx][-1][1])
        elif model=='natsql':
            list.append(all_preds[i][idx][0]['score'])
        else:
            print('Unknown model')

    std = np.std(list)
    return std


if __name__ == '__main__':
    id_label_map = {}

    name = 'DIRED4SP/preprocessing/preprocess_raw/parser_resdnatsql/resdsql_mc_preds/pred-beam-mc'
    preds_w_logit_n_score = []
    for i in range(0, 10):
        name1 = f'{name}-{i}.json' # Adjust this based on prediction file.
        print(name1)
        with open(name1) as f:
            preds_w_logit_n_score.append(json.load(f))

 
    with open('DIR/ED4SP/preprocessing/datasets/resdnatsql/ed_resdnatsql_beam_test.json') as f:
        legit_preds = json.load(f)
        
    legit_dev_ids = []
    for pred in legit_preds:
        if pred[0]['valid_sql'] != 1:
            print(pred[0]['sql'])
            continue
        legit_dev_ids.append(pred[0]['idx'])
        id_label_map[pred[0]['idx']] = pred[0]['label']

    N_samples = len(preds_w_logit_n_score[0])
    mc_logit_eval_json = []
    N = 0
    for i in range(N_samples):
        if i in legit_dev_ids:
            mc_logit_eval_json.append([{
                'id': N,
                'score': get_std(preds_w_logit_n_score, i, model='resdnatsql'),
                'label': id_label_map[i]
            }])
            N += 1

    print(len(mc_logit_eval_json))

    with open('eval_resdnatsql_mc_logit.json', 'w') as f:
        json.dump(mc_logit_eval_json, f, indent=2)

