import json
import numpy as np
from scipy import special
def main():
    empty_list = []
    filenames = [
        # 'parser_smbop/smbop_pred_dev.json',
        # 'parser_smbop/split05_beam_w_logit.json',
        # 'parser_smbop/split05_comp_beam_w_logit.json'
        # 'bridge_beam_test_w_logits.json',
        # 'parser_smbop/smbop_kaggledbqa/released_checkpoint_kaggle_beam_w_logits.json'
        
        # 'parser_bridge/bridge_beam_05_w_logits.json',
        # 'parser_bridge/bridge_beam_05_comp_w_logits.json',
        # 'parser_bridge/bridge_beam_kaggle_w_logits.json'
        

        # 'parser_natsql/resdnatsql_beam_dev_w_logits.json'
        # 'parser_natsql/natsql_beam_train_05_w_logits.json',
        # 'parser_natsql/natsql_beam_train_05_comp_w_logits.json',


        # 'parser_resdnatsql/resdnatsql_beam_dev_w_logits.json'
        'parser_resdnatsql/resdnatsql_beam_train_05_w_logits.json',
        'parser_resdnatsql/resdnatsql_beam_train_05_comp_w_logits.json',
        
    ]
    for filename in filenames:
        with open(filename) as f:
            data_json = json.load(f)
        pred_with_score = []
        
        for idx, raw_instance in enumerate(data_json):
            instance = []
            for _instance in raw_instance:
                if _instance != '':
                    instance.append(_instance)
            instance_pred_with_score = []
            if len(instance) > 0:

                    
                
                if 'smbop' in filename:
                    logits = np.array([entry[1] for entry in instance])
                    scores = special.softmax(logits)
                    scores = np.round(scores, 5)
                else:
                    scores = np.array([np.exp(entry[1]) for entry in instance])
                
                for i, pred in enumerate(instance):
                    if pred[0].lower().startswith('select'):
                        instance_pred_with_score.append({
                            'pred': pred[0],
                            'score': scores[i],
                            'idx': idx
                        })
                # print(instance_pred_with_score)
                pred_with_score.append(instance_pred_with_score)
            else:
                empty_list.append(idx)
        file_name = filename.split('.')[-2]
        print(f'{len(empty_list)} empty predictions')
        print(empty_list)
        print(len(pred_with_score), ' instances remained')
        with open(f'{file_name}.json'.replace('logits', 'score'), 'w') as f:
            json.dump(pred_with_score, f, indent=2)
        
            
    pass

if __name__ == '__main__':
    main()