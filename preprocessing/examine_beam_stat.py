'''
Scripts for producing dataset statistics.
'''

import json

def examine_beam(filename):
    print(filename)
    with open(filename) as f:
        data = json.load(f)
    total_hits = 0
    total_hits_ori = 0
    total_samples = 0
    total_disagreement = 0
    for beam in data:
        if 'test' in filename:
            total_hits += beam[0]['label']
            total_hits_ori += beam[0]['exec']
            total_samples += 1
            total_disagreement += 1 if beam[0]['label'] != beam[0]['exec'] else 0
        else:
            for b in beam:
                total_hits += b['label']
            total_samples += len(beam)
    
    print('# beam: ', len(data))
    print('# hits: ', total_hits)
    print('# hits_ori: ', total_hits_ori)
    print('# disagreement: ', total_disagreement)
    print('# miss: ', total_samples - total_hits)
    print('# total: ', total_samples)


filenames = [
    # 'datasets/smbop/ed_smbop_beam_train_sim2.json',
    # 'datasets/smbop/ed_smbop_beam_dev_sim2.json',
    # 'datasets/smbop/ed_smbop_beam_test_sim2.json',
    # 'datasets/natsql/ed_natsql_beam_train_sim2.json',
    # 'datasets/natsql/ed_natsql_beam_dev_sim2.json',
    # 'datasets/natsql/ed_natsql_beam_test_sim2.json',
    # 'datasets/resdnatsql/ed_resdnatsql_beam_test_sim2.json',
    # 'datasets/resdnatsql/ed_resdnatsql_beam_train_sim2.json',
    # 'datasets/resdnatsql/ed_resdnatsql_beam_dev_sim2.json'
]

for filename in filenames:
    examine_beam(filename)
