import json
import numpy as np
# with open('ed_released_checkpoint_dev_w_scores_dd.json') as f:
# with open('parser_smbop/smbop_pred_dev_w_score_exem2.json') as f:
# with open('parser_bridge/bridge_beam_test_w_score_exem2.json') as f:
# with open('parser_bridge/bridge_beam_kaggle_w_score_exem2.json') as f:
# with open('parser_natsql/natsql_beam_dev_exem2.json') as f:
# with open('parser_resdsql/resdsql_beam_dev_w_score_exem2.json') as f:
with open('parser_resdnatsql/resdnatsql_beam_dev_w_score_exem2.json') as f:
    test_json = json.load(f)
eval_json = []
N = 0
for idx, sample in enumerate(test_json):
    if sample[0]['valid_sql'] != 1:
        print(sample[0]['sql'])
        continue
    eval_json.append([{
        'id': N,
        'score': ex['confidence'],
        'label': ex['label']
    } for ex in sample])
    N += 1
    
print(len(eval_json))
with open('eval_resdsql_prob.json', 'w') as f:
    json.dump(eval_json, f, indent=2)