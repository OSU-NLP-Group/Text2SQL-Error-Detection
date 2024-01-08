import json
from collections import OrderedDict
import nltk
import sys
sys.path.append('../')
from tqdm import tqdm
def analyze_beam(beams):
    empty_beam = 0
    top_hit = 0
    beam_hit = 0
    beam_sizes = {}
    hit_nums = {}
    hit_miss = {}
    new_beams = []
    for beam in tqdm(beams):
        hits = 0
        misses = 0
        new_beam = {
            'spider_idx': beam[0]['idx'],
            'question': beam[0]['question'],
            'gold': beam[0]['gold'],
            'top_hit': 0,
            'beam_hit': 0,
            'beam': [(b['sql'], b['label']) for b in beam],

        }
        l = len(beam)
        score_key = 'label'
        # score_key = 'exec'
        if beam[0][score_key] == 1:
            top_hit += 1
            new_beam['top_hit'] = 1
        for b in beam:
            if b[score_key] == 1:
                hits += 1
            else:
                misses += 1
        if hits > 0:
            beam_hit += 1
        
        new_beam['beam_hit'] = 1 if hits > 0 else 0
        new_beam['beam_hits'] = hits
        new_beam['beam_misses'] = misses

        if l in beam_sizes:
            beam_sizes[l] += 1
        else:
            beam_sizes[l] = 1
        
        if hits in hit_nums:
            hit_nums[hits] += 1
        else:
            hit_nums[hits] = 1
        hm = str((hits, misses))
        if hm in hit_miss:
            hit_miss[hm] += 1
        else:
            hit_miss[hm] = 1

        new_beams.append(new_beam)
        
    beam_sizes = OrderedDict(sorted(beam_sizes.items()))
    
    print(f'{empty_beam} empty beams')
    print('beam sizes: ')
    print(json.dumps(dict(sorted(beam_sizes.items())), indent=2))
    print('beam hit nums: ')
    print(json.dumps(dict(sorted(hit_nums.items())), indent=2))
    print('(hit, miss): ')
    print(json.dumps(dict(sorted(hit_miss.items())), indent=2))
    print(f'Acc: {top_hit}/{len(beams)} = {round(top_hit/len(beams)*100, 2)}')
    print(f'Beam Hit: {beam_hit}/{len(beams)} = {round(beam_hit/len(beams)*100, 2)}')
    print('-' * 20)
    return new_beams


def deduplicate_beam(pred_beams):
    new_beams = []
    for beam in pred_beams:
        
        new_beam = []
        seen_sqls = []
        for b in beam:
            sql = ' '.join(nltk.word_tokenize(b['sql'].lower()))
            if sql in seen_sqls:
                continue
            seen_sqls.append(sql)
            x = b
            x['sql'] = sql
            new_beam.append(x)
        new_beams.append(new_beam)
    return new_beams


if __name__ == '__main__':
    filenames = [
        'parser_resdnatsql/resdnatsql_beam_dev_w_score_exem2.json',
        # 'parser_natsql/natsql_beam_dev_exem2.json',
        # 'ed_natsql_beam_05_exem.json',
        # 'ed_natsql_beam_01_dd.json',
        # 'ed_natsql_beam_03_dd.json',
        # 'ed_natsql_beam_05_dd.json'
    ]

    for filename in filenames:
        with open(filename) as fp:
            pred_beams = json.load(fp)

        new_beams = analyze_beam(pred_beams)
        
        # with open(filename.replace('.json', '_beam_view.json'), 'w') as fp:
            # json.dump(new_beams, fp, indent=2)
     
    # filenames = [
    #     'ed_natsql_beam_dev.json',
    #     'ed_natsql_beam_01.json',
    #     'ed_natsql_beam_03.json',
    #     'ed_natsql_beam_05.json'
    #     ]

    # for filename in filenames:
    #     with open(filename) as fp:
    #         pred_beams = json.load(fp)

    # #     new_beams = analyze_beam(pred_beams)
        
    # #     with open(filename.replace('.json', '_beams.json'), 'w') as fp:
    # #         json.dump(new_beams, fp, indent=2)
     
    #     new_beams = deduplicate_beam(pred_beams)
    #     with open(filename.replace('.json', '_dd.json'), 'w') as fp:
    #         json.dump(new_beams, fp, indent=2)
