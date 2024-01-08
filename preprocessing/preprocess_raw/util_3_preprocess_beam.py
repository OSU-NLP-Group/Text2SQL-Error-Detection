from io import StringIO
import json
import sys
from disamb_sql import disambiguate_items
from process_sql import tokenize
from spider_evaluation import build_foreign_key_map_from_json
from spider_utils import fixed_get_exec_score, get_exec_score, eval_exact_match
from tqdm import tqdm
from antlr4 import *
from SQLite_Parser.SQLiteLexer import SQLiteLexer
from SQLite_Parser.SQLiteParser import SQLiteParser
from func_timeout import func_timeout, FunctionTimedOut

from multiprocessing import Pool
def preprocess_data(data_json, ref_json, verbose=False, keep_top1=False, reverse=False):
    empty_em_ex_disagreement = 0
    new_data_list = []
    num_skipped_instance = 0
    kmaps = build_foreign_key_map_from_json('preprocess_raw/spider/tables.json')
    # kmaps = build_foreign_key_map_from_json('preprocess_raw/KaggleDBQA/KaggleDBQA_tables.json')
    idx = -1
    empty_gold_idx = []
    tables_file = 'preprocess_raw/spider/tables.json'
    # tables_file='preprocess_raw/KaggleDBQA/KaggleDBQA_tables.json'
    fixed_ori_exec_disagreement = 0
    assert len(data_json) <= len(ref_json)
    for beam in tqdm(data_json):
        idx += 1
        _instances = []
        new_instance = []
        if reverse:
            beam.reverse()
        seen_sqls = []
        for entry in beam:
            pred_sql = entry['pred']
            if pred_sql not in seen_sqls:
                _instances.append(entry)
                seen_sqls.append(pred_sql)
            else:
                continue
            idx = entry['idx']
            db_id = ref_json[idx]['db_id']
            try:
                sql_tokens = tokenize(pred_sql)
                sql_tokens = disambiguate_items(db_id, sql_tokens, tables_file, allow_aliases=False)
            except Exception as e:
                if verbose:
                    print(e)
                    print('idx: ', idx)
                    print('SQL: ', pred_sql)
                    print('db_id: ', db_id)
                    print('SQL Disambiguition Error')
                continue
            

        for entry in _instances:
            empty_gold = 0
            id = entry['idx']
            db_id = ref_json[id]['db_id']
            gold_sql = ref_json[id]['query']
            pred_sql = entry['pred']
                
            s = StringIO()
            stderr = sys.stderr
            sys.stderr = s
            
            lexer = SQLiteLexer(InputStream(pred_sql))
            stream = CommonTokenStream(lexer)
            parser = SQLiteParser(stream)
            tree = parser.select_stmt()
            s.seek(0)
            str = s.read()
            if str.startswith('line'):
                if verbose:
                    print('parsing err')
                sys.stderr = stderr
                continue
            sys.stderr = stderr
            exec_err=False
            msg = 'fail'
            
            try:
                exec_correct, msg = func_timeout(10, get_exec_score, args=(db_id, pred_sql, gold_sql, kmaps, verbose))
                fixed_exec_correct, msg = func_timeout(10, fixed_get_exec_score, args=(db_id, pred_sql, gold_sql, kmaps, verbose))
            except FunctionTimedOut as e:
                if verbose:
                    print('exec timed out')
                exec_correct=False
                fixed_exec_correct=False
                exec_err=True
                # continue
            except Exception as e:
                if verbose:
                    print('Exception: ', e)
                exec_correct=False
                fixed_exec_correct=False
                exec_err=True
                # continue
            if msg == 'gold error':
                continue
            
            if msg == 'empty gold':
                empty_gold = 1
                empty_gold_idx.append(idx)
            
            try:
                em_score = eval_exact_match(db_id, pred_sql, gold_sql, kmaps)
            except Exception:
                em_score=0
            exec_socre = 1 if exec_correct else 0 
            fixed_exec_score = 1 if fixed_exec_correct else 0 
            final_score = fixed_exec_score
            if empty_gold:
                final_score = em_score
                if fixed_exec_score != em_score:
                    empty_em_ex_disagreement += 1
            if fixed_exec_score != exec_socre:
                exec_disagree = True
                fixed_ori_exec_disagreement += 1
            else:
                exec_disagree = False
            new_entry = {
                'idx': entry['idx'],
                'db_id': db_id,
                'question': ref_json[id]['question'],
                'confidence': entry['score'],
                'sql': entry['pred'],
                'gold': gold_sql,
                'valid_sql': 0 if exec_err or msg not in ['success', 'empty gold', 'number of result units unmatched'] else 1,
                'empty_gold': empty_gold,
                'exec': exec_socre,
                'fixed_exec': fixed_exec_score,
                'exec_disagree': 1 if exec_disagree else 0,
                'em': em_score,
                'label': final_score
            }
            if new_entry['valid_sql'] == 0:
                continue
            new_instance.append(new_entry)
            if keep_top1:
                break
            elif len(new_instance) == 5:
                break
        if len(new_instance)==0:
            num_skipped_instance += 1
            continue
        new_data_list.append(new_instance)
    
    print(f'processing finished, skipped{num_skipped_instance} instances.')
    print(len(list(set(empty_gold_idx))), ' empty golds')
    print('#Empty golds: ', empty_gold_idx)
    print('#Empty ex em disagreement: ', empty_em_ex_disagreement)
    
    return new_data_list
            
def main():

    file_names = [

        # 'parser_natsql/natsql_beam_dev.json',
        # 'parser_natsql/natsql_beam_05.json',
        # 'parser_natsql/natsql_beam_05_comp.json',
        # 'parser_natsql/natsql_beam_spider_train.json',

        # 'parser_smbop/smbop_pred_dev_w_score.json',
        # 'parser_smbop/split05_beam_w_score.json',
        # 'parser_smbop/split05_comp_beam_w_score.json'

        # 'parser_bridge/bridge_beam_test_w_score.json',
        # 'parser_bridge/bridge_beam_05_w_score.json',
        # 'parser_bridge/bridge_beam_05_comp_w_score.json',

        # 'parser_bridge/bridge_beam_kaggle_w_score.json'
        # 'parser_resdnatsql/resdnatsql_beam_dev_w_score.json',
        # 'parser_resdnatsql/resdnatsql_beam_train_05_w_score.json',
        'parser_resdnatsql/resdnatsql_beam_train_05_comp_w_score.json',
        
    ]
    ref_names = [
        # 'spider/spider_train_split_0.5.json',
        'spider/spider_train_split_0.5_comp.json',
        # 'KaggleDBQA/KaggleDBQA_test.json'
        
    ]
    for filename, ref in zip(file_names, ref_names):
        print(filename)
        with open(filename, 'r') as f:
            data_json = json.load(f)
        with open(ref, 'r') as f:
            ref_json = json.load(f)
        keep_top1 = False
        processed_data = preprocess_data(data_json, ref_json, verbose=True, keep_top1=keep_top1, reverse=False)
        if 'diagnose' in filename:
            continue

        
        with open(filename.replace('.json', '_exem2.json'), 'w') as f:
            json.dump(processed_data, f, indent=2)

if __name__ == '__main__':
    main()