from evaluation import *
import json
def get_hardness(sql_str, db_id):
    db = os.path.join('/data/data/spider/database/', db_id, db_id+'.sqlite')
    schema = Schema(get_schema(db))
    sql = get_sql(schema, sql_str)
    hardness = eval_hardness(sql)
    if hardness == 'easy':
        hardness = eval_hardness(sql, verbose=True)
    return hardness
def eval_hardness(sql, verbose = False):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)
    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
        count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
    ):
        return "medium"
    elif (
        (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
        or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
        or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
    ):
        return "hard"
    else:
        return "extra"