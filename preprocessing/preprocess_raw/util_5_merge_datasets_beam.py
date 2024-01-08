import json


data_file_lists = [
    # ('parser_natsql/natsql_beam_05_comp_train_ub_fb.json', 'split05_comp'),
    # ('parser_natsql/natsql_beam_05_train_ub_fb.json', 'split05'),
    # ('parser_smbop/smbop_beam_05_train_ub_fb.json', 'split05'),
    # ('parser_smbop/smbop_beam_05_comp_train_ub_fb.json', 'split05_comp'),
    # ('parser_bridge/bridge_beam_05_dev_ub_fb.json', 'split05'),
    # ('parser_bridge/bridge_beam_05_comp_dev_ub_fb_json', 'split05_comp'),
    ('parser_resdnatsql/resdnatsql_beam_dev_05_ub_fb.json', 'split05'),
    ('parser_resdnatsql/resdnatsql_beam_dev_05_comp_ub_fb.json', 'split05_comp'),
    
]

new_data = []
seen_pairs = []
n_dup = 0
dup_sqls = []
for filename, split in data_file_lists:
    with open(filename) as f:
        data = json.load(f)
    new_data.extend(data)

with open('ed_resdnatsql_full_beam_dev.json', 'w') as f:
    json.dump(new_data, f, indent=2)