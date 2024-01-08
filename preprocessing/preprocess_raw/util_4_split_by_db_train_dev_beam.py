import copy
import json
import os
import random
random.seed(0)
def get_samples(beam, config ):

    pos_found = False
    neg_found = False
    pos_samples = []
    neg_samples = []
    beam_smaples = []
    for sample in beam:
        if sample['valid_sql'] != 1:
            continue
        
        if sample['label'] == 0:
            if config['top1'] and neg_found:
                continue
            new_neg_sample = sample
            new_neg_sample['source'] = 'parser'
            neg_samples.append(new_neg_sample)
            beam_smaples.append(new_neg_sample)
            neg_found = True
        if sample['label'] == 1:
            new_pos_sample = sample
            new_pos_sample['source'] = 'parser'
            pos_samples.append(new_pos_sample)
            beam_smaples.append(new_pos_sample)
            pos_found = True
        if config['top1']:
            if pos_found and neg_found:
                break
    

    return beam_smaples, pos_found, neg_found
    
def group_samples_by_db(samples, config):
    samples_by_db = {}
    num_samples = 0
    for beam in samples:
        beam_samples, pos_found, neg_found = get_samples(beam, config)
        db_id = beam[0]['db_id']
        if len(beam_samples) == 0:
            continue
        if config['both_labels']:
            if pos_found and neg_found:
                if db_id in samples_by_db:
                    samples_by_db[db_id].append(beam_samples)
                else:
                    samples_by_db[db_id]=[beam_samples]
                num_samples += 1
        else:
            if db_id in samples_by_db:
                samples_by_db[db_id].append(beam_samples)
            else:
                samples_by_db[db_id]=[beam_samples]
            num_samples += 1

    return samples_by_db, num_samples


def create_train_dev(samples, config, train_split_ratio=0.8):
    samples_by_db, N_samples = group_samples_by_db(samples, config)
    db_ids = list(samples_by_db.keys())
    train_split = []
    dev_split = []
        
    # if os.path.exists('ori_devset_db_ids.json'):
        # with open('ori_devset_db_ids.json') as f:
    if os.path.exists('devset_db_ids.json'):
        with open('devset_db_ids.json') as f:
            dev_db_ids = json.load(f)
        print('dev db_ids: ', dev_db_ids)
        for db_id in db_ids:
            if db_id in dev_db_ids:
                dev_split.extend(samples_by_db[db_id])
            else:
                train_split.extend(samples_by_db[db_id])
    else:
        while(len(train_split) < N_samples * train_split_ratio):
            db_id = random.sample(db_ids, 1)[0]
            db_ids.remove(db_id)
            train_split.extend(samples_by_db[db_id])
        for db_id in db_ids:
            dev_split.extend(samples_by_db[db_id])
    return train_split, dev_split, db_ids

def main():
    data_filenames = [
        # 'parser_natsql/natsql_beam_05_exem2',
        # 'parser_natsql/natsql_beam_05_comp_exem2',
        # 'parser_smbop/split05_comp_beam_w_score_exem2',
        # 'parser_smbop/split05_beam_w_score_exem2',
        # 'parser_bridge/bridge_beam_05_w_score_exem2',
        # 'parser_bridge/bridge_beam_05_comp_w_score_exem2',
        # 'parser_natsql/natsql_beam_spider_train_exem2'
        'parser_resdnatsql/resdnatsql_beam_train_05_comp_w_score_exem2',
        'parser_resdnatsql/resdnatsql_beam_train_05_w_score_exem2'
    ]
    # Configs for preliminary experiments.
    config = {
        'top1': False, # Only keep the top positive or negative prediction (if any) of each beam
        'both_labels': False # Requires each beam to have both positive and negative samples. If false, there could be beams with only positive or negative samples
    }

    if not config['both_labels']:
        suffix += '_ub' # ub -> unbalanced
    if not config['top1']:
        suffix += '_fb' # fb -> full beam
    for filename in data_filenames:
        print('processing ', filename)
        with open(f'{filename}.json') as f:
            samples = json.load(f)
        train_samples, dev_samples, dev_db_ids = create_train_dev(samples, config)
        print('train: ', len(train_samples))
        print('dev: ', len(dev_samples))
        with open(f'{filename}_train{suffix}.json'.replace('_w_score', '').replace('_exem2', ''), 'w') as f:
            json.dump(train_samples, f, indent=2)
        with open(f'{filename}_dev{suffix}.json'.replace('_w_score', '').replace('_exem2', ''), 'w') as f:
            json.dump(dev_samples, f, indent=2)
        # with open(f'parser_natsql/natsql_dev_db_ids.json', 'w') as f:
            # json.dump(dev_db_ids, f)
        

        



if __name__ == '__main__':
    main()