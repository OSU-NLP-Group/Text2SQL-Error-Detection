import json
import os
import numpy as np

def compute_reranking_metrics(expr_name, threshold=0.5, dump_reranked=False):

    dataset_name = DATASET_NAME

    eval_filename = f'{expr_name}/eval_{dataset_name}_{setting}.json'
    with open(eval_filename) as f:
        eval_json = json.load(f)
    

    reranked_beams = []

    acc = 0
    reranked_acc_ed = 0
    reranked_acc_ot = 0
    ed_improve = 0
    ed_hurt = 0
    ot_improve = 0
    ot_hurt = 0
    beam_hit = 0
    improved_id = []
    hurted_id = []
    for beam in eval_json:
        acc += beam[0]['label']
        oracle_beam = sorted(beam, key = lambda x:x['label'], reverse=True)
        beam_hit += oracle_beam[0]['label']
        if beam[0]['score'] > threshold:
            reranked_beam_by_ed = beam

        else:
            reranked_beam_by_ed= sorted(beam, key = lambda x:x['score'], reverse=True)

        
        if reranked_beam_by_ed[0]['score'] > threshold:
            reranked_beams.append(reranked_beam_by_ed)
        else:
            reranked_beams.append(beam)

        # reranked_beams.append(reranked_beam_by_ed)
        reranked_acc_ed += reranked_beam_by_ed[0]['label']
        if beam[0]['label'] - reranked_beam_by_ed[0]['label'] < 0:
            ed_improve += 1
            improved_id.append(beam[0]['id'])
        elif beam[0]['label'] - reranked_beam_by_ed[0]['label'] > 0:
            ed_hurt += 1
            hurted_id.append(beam[0]['id'])
    
    if dump_reranked:
        with open(f'{expr_name}/eval_{dataset_name}_{setting}_rr2.json', 'w') as f:
            json.dump(reranked_beams, f, indent=2)

    N = len(eval_json)

    return reranked_acc_ed



# DATASET_NAME = 'smbop'
# DATASET_NAME = 'natsql'
DATASET_NAME = 'resdnatsql'
# DATASET_NAME = 'bridge_kaggle'

setting = 'beam'
def main():
    
    exprs = [
        ('NatSQL/CodeBERT', 1),
        ('NatSQL/CodeBERT_GAT', 1),
        ('ResdNatSQL/CodeBERT', 1),
        ('ResdNatSQL/CodeBERT_GAT', 1),
        ('SmBoP/CodeBERT', 1),
        ('SmBoP/CodeBERT_GAT', 1),

    ]
    N_samples = 1034 #Spider
    # N_samples = 366 # kaggleDBQA
    for expr in exprs:
        expr_name = expr[0]
        rr_all_accs = []
        rr_ed_accs = []
        for i in range(expr[1]):
            expr_name = f'{expr[0]}_{i+1}'
            rr_acc = compute_reranking_metrics(expr_name, threshold=1.0)
            rr_ed_acc = compute_reranking_metrics(expr_name, threshold=0.5, dump_reranked=True)
            rr_all_accs.append(rr_acc/N_samples*100)
            rr_ed_accs.append(rr_ed_acc/N_samples*100)
        print(expr[0])
        print(f'average reranking-all: {np.round(np.mean(rr_all_accs),3)}({np.round(np.std(rr_all_accs),3)})')
        print(f'average reranking-ed: {np.round(np.mean(rr_ed_accs),3)}({np.round(np.std(rr_ed_accs),3)})')
        

if __name__ == '__main__':
    main()