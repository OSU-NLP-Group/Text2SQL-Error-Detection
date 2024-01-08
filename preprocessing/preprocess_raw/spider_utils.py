import mo_sql_parsing
import sqlite3
from spider.evaluation import build_foreign_key_map_from_json, build_valid_col_units, eval_IUEN, eval_and_or, eval_group, eval_having, eval_keywords, eval_order, eval_sel, eval_where, get_scores, rebuild_sql_col, rebuild_sql_val

from process_sql import Schema, get_schema, get_sql
# DB_PATH = '/data/data/spider/database' # for use in docker
DB_PATH = '/research/nfs_su_809/chen.10216/projects/data/spider/database'

def fetch_all_utf8(cursor):
    # return cursor.fetchall()
    q_res = []
    q_r = ("dummy")
    while q_r is not None:
        try:
            q_r = cursor.fetchone()
        except:
            continue

        if q_r is not None:
            q_res.append(q_r)
    return q_res
        
def get_exec_score(db_id, pred_sql, gold_sql, kmaps,verbose=False):
    # mostly copied from evaluation.py of Spider's repo
    msg = ''
    db_path = f'{DB_PATH}/{db_id}/{db_id}.sqlite'
    schema=Schema(get_schema(db_path))

    try:
        g_sql = get_sql(schema, gold_sql)
    except:
        msg = 'gold error'
        return False, msg
    
    try:
        p_sql = get_sql(schema, pred_sql)
    except:
        # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
        p_sql = {
        "except": None,
        "from": {
            "conds": [],
            "table_units": []
        },
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [
            False,
            []
        ],
        "union": None,
        "where": []
        }
        msg='fail'

    
    kmap = kmaps[db_id]
    g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
    
    conn = sqlite3.connect(db_path)
    # conn.text_factory = bytes
    cursor = conn.cursor()
    msg = 'success'
    
    try:
        
        cursor.execute(gold_sql)
        # gold_exec = cursor.fetchall()
        gold_exec = fetch_all_utf8(cursor)
        gold_result = tuple(gold_exec)
        cursor.execute(pred_sql)
        # pred_result = tuple(cursor.fetchall())
        pred_exec = fetch_all_utf8(cursor)
        pred_result = tuple(pred_exec)
        cursor.close()
    except Exception as e:
        msg = 'fail'
        if verbose:
            print('ori SQLite Exec Error - ', e)
            print('db: ', db_id)
            print('gold: ', gold_sql)
            print('pred: ', pred_sql)
        return False, msg
    finally:
        conn.close()


    def res_map(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            rmap[key] = [r[idx] for r in res]
        return rmap

    pred_val_units = [unit[1] for unit in p_sql['select'][1]]
    gold_val_units = [unit[1] for unit in g_sql['select'][1]]
    if len(gold_exec) == 0:
        msg = 'empty gold'
    return res_map(pred_result, pred_val_units) == res_map(gold_result, gold_val_units), msg

def fixed_get_exec_score(db_id, pred_sql, gold_sql, kmaps,verbose=False):
    # mostly copied from evaluation.py of Spider's repo
    msg = ''
    db_path = f'{DB_PATH}/{db_id}/{db_id}.sqlite'
    
    schema=Schema(get_schema(db_path))
    try:
        g_sql = get_sql(schema, gold_sql)
    except:
        msg = 'gold error'
        return False, msg

    try:
        p_sql = get_sql(schema, pred_sql)
    except:
        # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
        p_sql = {
        "except": None,
        "from": {
            "conds": [],
            "table_units": []
        },
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [
            False,
            []
        ],
        "union": None,
        "where": []
        }
        msg='fail'
    
    # fix bug caused by 'limit 1' when multiple aggregated results with the same value are ordered differently
    conn = sqlite3.connect(db_path)
    # conn.text_factory = bytes
    cursor = conn.cursor()
    msg = 'success'
    ignore_order = False
    try:
        if p_sql['limit'] == g_sql['limit'] and p_sql['limit'] is not None:
            # remove the limit clauses if they are identical
            p_sql['limit'] == None
            g_sql['limit'] == None
            p_sql_1 = mo_sql_parsing.parse_mysql(pred_sql)
            g_sql_1 = mo_sql_parsing.parse_mysql(gold_sql)
            if 'limit' in p_sql_1 and 'limit' in g_sql_1:
                p_sql_1.pop('limit')
                pred_sql = mo_sql_parsing.format(p_sql_1)
                
                g_sql_1.pop('limit')
                gold_sql = mo_sql_parsing.format(g_sql_1)
        
        kmap = kmaps[db_id]
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        
        cursor.execute(gold_sql)
        gold_exec = fetch_all_utf8(cursor)
        gold_result = tuple(gold_exec)
        cursor.execute(pred_sql)
        pred_exec = fetch_all_utf8(cursor)
        pred_result = tuple(pred_exec)
        cursor.close()

        # fix bug caused by considering order unnecessarily
        if p_sql['orderBy'] == g_sql['orderBy'] and len(gold_exec) == len(pred_exec):
            ignore_order = True
    except sqlite3.Error as error:
        msg = 'fail'
        if verbose:
            print('fixed SQLite Exec Error - ', error)
            print('db: ', db_id)
            print('gold: ', gold_sql)
            print('pred: ', pred_sql)
    except Exception as e:
        msg = 'fail'
        if verbose:
            print('Exception: ', e)
    finally:
        conn.close()
    if msg == 'fail':
        return False, msg
    def res_map(res, val_units, ignore_order=False):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            agg_id, col_unit = val_unit
            key = tuple(col_unit[1]) if not col_unit[2] else (col_unit[0], tuple(col_unit[1]), tuple(col_unit[2]))
            key = (agg_id, key)
            if ignore_order:
                try:
                    rmap[key] = sorted([r[idx] for r in res])
                except:
                    rmap[key] = [r[idx] for r in res]
            else:
                rmap[key] = [r[idx] for r in res]
        return rmap

    
    # Add aggregation operator id into column identifier
    pred_val_units = [(unit[0], unit[1]) for unit in p_sql['select'][1]]
    gold_val_units = [(unit[0], unit[1]) for unit in g_sql['select'][1]]

    if len(pred_val_units) != len(gold_val_units):
        label = 0
        msg = 'number of result units unmatched'
        return label, msg
    else:
        pred_res_map = res_map(pred_result, pred_val_units, ignore_order)
        gold_res_map = res_map(gold_result, gold_val_units, ignore_order)
        label = gold_res_map == pred_res_map

    if len(gold_exec) == 0:
        msg = 'empty gold'
    return label, msg
    
def eval_exact_match(db_id, pred_sql, gold_sql, kmaps,verbose=False):
    # mostly copied from evaluation.py of Spider's repo
    msg = ''
    db_path = f'{DB_PATH}/{db_id}/{db_id}.sqlite'
    schema=Schema(get_schema(db_path))
    g_sql = get_sql(schema, gold_sql)
    
    
    try:
        p_sql = get_sql(schema, pred_sql)
    except:
        # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
        p_sql = {
        "except": None,
        "from": {
            "conds": [],
            "table_units": []
        },
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [
            False,
            []
        ],
        "union": None,
        "where": []
        }
    
    if p_sql['limit'] == g_sql['limit']:
        p_sql['limit'] == None
        g_sql['limit'] == None
    kmap = kmaps[db_id]
    g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
    
    partial_scores = eval_partial_match(p_sql, g_sql)

    for _, score in partial_scores.items():
        if score['f1'] != 1:
            return 0
    if len(g_sql['from']['table_units']) > 0:
        label_tables = sorted(g_sql['from']['table_units'])
        pred_tables = sorted(p_sql['from']['table_units'])
        return 1 if label_tables == pred_tables else 0
    return 1

def eval_partial_match(pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        return res

def main():
    # A test case for fixed evaluation script
    kmaps = build_foreign_key_map_from_json('spider/tables.json')
    db_id = 'car_1'
    pred_sql = 'select countries.countryname from countries where countries.countryid not in ( select car_makers.country from car_makers )'
    gold_sql = "SELECT CountryName FROM countries EXCEPT SELECT T1.CountryName FROM countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.countryId  =  T2.Country;"

    db_id = 'flight_2'
    pred_sql = 'select airlines.abbreviation , airlines.country from airlines join flights on airlines.uid = flights.airline group by flights.airline order by count ( * ) asc limit 1'
    gold_sql = 'select airlines.abbreviation , airlines.country from airlines join flights on airlines.uid = flights.airline group by airlines.airline order by count ( * ) limit 1'
    db_id='tvshow'

    pred_sql = 'select min ( tv_series.share ) , max ( tv_series.share ), min ( tv_series.share ) from tv_series'
    gold_sql = 'select max ( tv_series.share ) , min ( tv_series.share ) from tv_series'

    print(get_exec_score(db_id, pred_sql, gold_sql, kmaps))
    print(fixed_get_exec_score(db_id, pred_sql, gold_sql, kmaps, verbose=True))
    # pass
if __name__ == '__main__':
    main()