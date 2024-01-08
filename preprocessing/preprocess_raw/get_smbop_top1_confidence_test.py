import json
import numpy as np
from scipy import special
def deduplicate_test_beam(file_name):
    with open(file_name+'.json') as f:
        test_preds = json.load(f)
    deduplicated_beams = []
    for idx, beam in enumerate(test_preds):
        seen_sqls = []
        new_beam = []
        
        for i in range(len(beam)-1, -1, -1):
            if beam[i] != '':
                if beam[i][0] not in seen_sqls:
                    new_beam.append(beam[i])
                    seen_sqls.append(beam[i][0])
        logits = np.array([entry[1] for entry in new_beam])
        scores = special.softmax(logits)
        scores = np.round(scores, 5)
        new_beam1 = []
        for i, entry in enumerate(new_beam):
            new_beam1.append({
                'idx': idx,
                'pred': entry[0],
                'logit': entry[1],
                'score': scores[i]
            })
        deduplicated_beams.append([new_beam1[0]])

    with open(file_name.replace('logit', 'score')+'_dd.json', 'w') as f:
        json.dump(deduplicated_beams, f, indent=2)
filenames = [
    # 'released_checkpoint_dev_beam_w_logit_mc_6',
    # 'released_checkpoint_dev_beam_w_logit_mc_7',
    # 'released_checkpoint_dev_beam_w_logit_mc_8',
    # 'released_checkpoint_dev_beam_w_logit_mc_9',
    # 'released_checkpoint_dev_beam_w_logit_mc_10',
    # 'parser_smbop/smbop_kaggledbqa/released_checkpoint_kaggle_beam_w_logits'
]
for file in filenames:
    deduplicate_test_beam(file)
