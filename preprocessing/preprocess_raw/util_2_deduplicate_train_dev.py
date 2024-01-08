import json

def deduplicate_beam(beam):
    # only keep 1 neg sample and up to 1 pos sample
    l = len(beam)
    new_beam = []
    seen_sqls = []

    for i in range(l-1, -1, -1):
        if beam[i]['pred'] not in seen_sqls:
            new_beam.append(beam[i])
            seen_sqls.append(beam[i]['pred'])
    return new_beam

def main():
    data_filenames = [
        # 'parser_smbop/smbop_pred_dev_w_score',
        # 'parser_smbop/split05_beam_w_score',
        # 'parser_smbop/split05_comp_beam_w_score_split05'
        # 'parser_bridge/bridge_beam_05_w_score',
        # 'parser_bridge/bridge_beam_05_comp_w_score',
        # 'parser_resdsql/resdsql_beam_dev_w_score',
        # 'parser_resdsql/resdsql_beam_05_w_score',
        # 'parser_resdsql/resdsql_beam_05_comp_w_score',
        'parser_resdnatsql/resdnatsql_beam_train_05_w_score',
        # 'parser_resdnatsql/resdnatsql_beam_train_05_comp_w_score',


    ]
    
    N_empty_beams = 0
    for filename in data_filenames:
        is_test_set = 'test' in filename
        with open(f'{filename}.json') as f:
            data_json = json.load(f)
        deduplicated_data = []
        for beam in data_json:
            if is_test_set:
                beam = beam[-1:]
            deduplicated_beam = deduplicate_beam(beam)

            if len(deduplicated_beam) == 0:
                N_empty_beams += 1
            deduplicated_data.append(deduplicated_beam)
        print(f'{filename} - N empty beams: {N_empty_beams}')
        with open(f'{filename}_dd.json', 'w') as f:
            json.dump(deduplicated_data, f, indent=2)

    # pass

if __name__ == '__main__':
    main()