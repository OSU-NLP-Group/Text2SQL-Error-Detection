import json
with open('ed_smbop_full_dev.json') as f:
    dev_json = json.load(f)

db_ids = list(set([beam['db_id'] for beam in dev_json]))
with open('ori_devset_db_ids.json', 'w') as f:
    json.dump(db_ids, f)
