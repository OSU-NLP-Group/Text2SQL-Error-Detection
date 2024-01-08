#%%
from transformers import RobertaTokenizer
import stanza 
import json
from tqdm import tqdm
from process_sql import get_sql, tokenize
from preprocess_raw.disamb_sql import disambiguate_items
from util_add_edges import add_global_edges, add_sequential_edges
import sys
from io import StringIO
from antlr4 import *
from SQLite_Parser.SQLiteLexer import SQLiteLexer
from SQLite_Parser.SQLiteParser import SQLiteParser
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # Set the following 3 settings to False for ablation study on parse tree simplification.
    parser.add_argument('--simplify_parse_trees', action='store_true', default=False)
    parser.add_argument('--merge_ast_dot', action='store_true', default=False)
    parser.add_argument('--remove_join_constraint', action='store_true', default=False)

    args = parser.parse_args()
    return args
args = parse_args()

codebert_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse,constituency',package={'constituency': 'wsj_bert'})
PUNCTS = ['.', ',', ';',':', '?', '!', '\'']
SQL_KEYWORDS_TO_IGNORE = [SQLiteParser.COMMA, SQLiteParser.OPEN_PAR, SQLiteParser.CLOSE_PAR]
if args.remove_join_constraint:
    SQL_KEYWORDS_TO_IGNORE.append(SQLiteParser.JOIN_)

def simplify_consti_tree(tree):
    '''
    Argument: A constituency tree node returned by Stanza
    '''
    if len(tree.children) == 0:
        return tree
    elif len(tree.children) == 1:
        return simplify_consti_tree(tree.children[0])
    else:
        new_children = []
        for child in tree.children:
            new_child = simplify_consti_tree(child)
            new_children.append(new_child)
        tree.children = new_children
        return tree

def visit_consti_tree(tree, level = 0, datapack = {}):
    if len(tree.children) == 0:
        # terminal nodes
        node_idx = datapack['t_node_count'] + 1
        datapack['t_node_count'] = node_idx
        datapack['t_nodes'][node_idx] = tree.label
        datapack['t_tokens'].append(tree.label)
        return node_idx, datapack
    else:
        # non-terminal nodes
        node_idx = datapack['nt_node_count'] + 1
        datapack['nt_node_count'] = node_idx
        datapack['nt_nodes'][node_idx] = tree.label
        for child in tree.children:            
            child_idx, datapack = visit_consti_tree(child, level=level+1, datapack=datapack)
            datapack['edges'].append([node_idx, child_idx])
            datapack['edges'].append([child_idx, node_idx])
        return node_idx, datapack

def parse_and_process_question(question:str, nlp, tokenizer):
    # question=question.lower()
    doc = nlp(question)
    sub_word_lens = []
    input_ids = []
    toks = []
    for sentence in doc.sentences[:1]: # only process 1 question for now
        
        N_words = len(sentence.words)-1
        datapack = {
            't_nodes':{},
            'nt_nodes': {},
            'edges': [],
            'dep_edges': [],
            'token2NodeMap': {},
            't_node_count': -1,
            'nt_node_count': N_words,
            't_tokens': [],
            'nl_input_tokens': [],
            'dep_root': -1
            }
        consti_tree = sentence.constituency
        if args.simplify_parse_trees:
            consti_tree = simplify_consti_tree(consti_tree)
        root_idx, datapack = visit_consti_tree(consti_tree, datapack=datapack)

        for word in sentence.words:
            text = word.text
            if word.head > 0:
                datapack['dep_edges'].append([word.id,word.head])
                datapack['dep_edges'].append([word.head,word.id])
            if word.head == 0:
                datapack['dep_root'] = word.id
            if word.id < len(sentence.words) and word.text not in PUNCTS and word.id > 1:
                text = ' ' + text
            toks.append(text)
            tokenized_word = tokenizer.tokenize(text)
            datapack['nl_input_tokens'].extend(tokenized_word)
            input_ids += (tokenizer.convert_tokens_to_ids(tokenized_word))
            sub_word_lens.append(len(tokenized_word))

        reconstructed_question = ' '.join(toks)
    return reconstructed_question, input_ids, sub_word_lens, root_idx, datapack


# %%
def parse_and_process_sql(sql:str, db_id: str, tables_file:str):
    try:
        sql_tokens1=None
        sql_tokens2=None
        sql_tokens3=None
        sql_tokens = tokenize(sql)
        sql_tokens1=' '.join(sql_tokens)
        sql_tokens = disambiguate_items(db_id, sql_tokens, tables_file, allow_aliases=False)
        sql_tokens2=' '.join(sql_tokens)
        sql_tokens = tokenize(sql_tokens)
        sql_tokens3=' '.join(sql_tokens)
        sql = ' '.join(sql_tokens)
        
    except Exception as e:
        print(e)
        print('SQL: ', sql)
        print('db_id: ', db_id)
        print('1-->', sql_tokens1)
        print('2-->', sql_tokens2)
        print('3-->', sql_tokens3)
        print('SQL Disambiguition Error')
        return {}, False

    s = StringIO()
    stderr = sys.stderr
    sys.stderr = s
    
    lexer = SQLiteLexer(InputStream(sql))
    stream = CommonTokenStream(lexer)
    parser = SQLiteParser(stream)
    tree = parser.select_stmt()
    s.seek(0)
    str = s.read()
    if str.startswith('line'):
        sys.stderr = stderr
        return {}, False
    sys.stderr = stderr
    if args.simplify_parse_trees:
        tree = CST_to_AST(tree)
    datapack = {
        't_nodes':{},
        'nt_nodes': {},
        'edges': [],
        'token2NodeMap': {},
        't_node_count': -1, # serves as index during ast traversal, and will be updated
        'nt_node_count': count_sql_t_nodes(tree)-1, # serves as index during ast traversal, and will be updated
        't_tokens': [],
        't_input_tokens':[],
        't_input_lens': [],
        't_input_ids': [],
        't_mask': [],
    }
    _, datapack =  visit_ast(tree, datapack=datapack)
    return datapack, True

def count_sql_t_nodes(tree):
    '''
    Count the number of terminal nodes.
    '''
    
    if tree.getChildCount() == 0:
        if tree.symbol.type not in SQL_KEYWORDS_TO_IGNORE:
            return 1    
        else:
            return 0
    elif args.merge_ast_dot and SQLiteParser.ruleNames[tree.getRuleIndex()] == 'expr' and len(tree.children) > 1 and tree.children[1].getText()=='.':
            # ['tab', '.', 'col'] -> ['tab.col']
            return 1
    else:
        num_nodes = 0
        for child in tree.getChildren():
            num_child_nodes = count_sql_t_nodes(child)
            num_nodes += num_child_nodes
        return num_nodes
def CST_to_AST(tree):
    '''
    Arguments: tree: a CST node returned by Antlr
    '''
    child_count = tree.getChildCount()
    # Putting removal of SQL_KEYWORDS_TO_IGNORE in visit_ast()
    if child_count == 0:
        return tree
    elif child_count == 1 and 'stmt' not in tree.getText() and 'clause' not in tree.getText():
        # remove non-terminal parse tree nodes with only 1 child
        return CST_to_AST(tree.children[0])
    elif args.remove_join_constraint and SQLiteParser.ruleNames[tree.getRuleIndex()] == 'join_constraint':
        return None
    else:
        new_children = []
        for child in tree.getChildren():
            new_child = CST_to_AST(child)
            if new_child is not None:
                new_children.append(new_child)
        tree.children = new_children
        return tree
def visit_ast(tree, level = 0, datapack = {}):
    if tree.getChildCount() == 0 or (tree.getChildCount()>=3 and args.merge_ast_dot and SQLiteParser.ruleNames[tree.getRuleIndex()] == 'expr' and tree.children[1].getText()=='.'):
        # terminal nodes
        node_idx = datapack['t_node_count']
        if args.merge_ast_dot and tree.getChildCount() > 0 and SQLiteParser.ruleNames[tree.getRuleIndex()] == 'expr' and len(tree.children) > 1 and tree.children[1].getText()=='.':
            # merge dot expr subtree into one terminal node
            # i.e. ['tab', '.', 'col'] -> ['tab.col']
            child_texts = []
            for c in tree.getChildren():
                child_texts.append(c.getText())
            node_text = ' '.join(child_texts)
            datapack['t_mask'].append(1)
            node_idx += 1
        else:
            node_text = tree.getText()
            if tree.symbol.type not in SQL_KEYWORDS_TO_IGNORE:
                node_idx += 1
                datapack['t_nodes'][node_idx] = tree.getText()
                datapack['t_mask'].append(1)
            else:
                datapack['t_mask'].append(0)
            
        if node_idx > 0:
            node_text = ' ' + node_text
        datapack['t_node_count'] = node_idx
        datapack['t_nodes'][node_idx] = node_text
        datapack['t_tokens'].append(node_text)
        tokenized_node_text = codebert_tokenizer.tokenize(node_text)
        token_ids = codebert_tokenizer.convert_tokens_to_ids(tokenized_node_text)
        datapack['t_input_tokens'] += tokenized_node_text
        datapack['t_input_lens'].append(len(token_ids))
        datapack['t_input_ids'] += (token_ids)
        return node_idx, datapack
    else:
        # non-terminal nodes
        # if args.remove_join_constraint:
            # assert SQLiteParser.ruleNames[tree.getRuleIndex()] != 'join_constraint'
        node_idx = datapack['nt_node_count'] + 1
        datapack['nt_node_count'] = node_idx
        datapack['nt_nodes'][node_idx] = SQLiteParser.ruleNames[tree.getRuleIndex()]
        for child in tree.getChildren():            
            if child.getChildCount() > 0 or (child.getChildCount()==0 and child.symbol.type not in SQL_KEYWORDS_TO_IGNORE):
                child_idx, datapack = visit_ast(child, level=level+1, datapack=datapack)
                datapack['edges'].append([node_idx, child_idx])
                datapack['edges'].append([child_idx, node_idx])
        return node_idx, datapack
def preprocess_samples(samples, tables_dict):
    new_dataset = []
    err_sql_count = 0
    for sample in samples:
        if sample['valid_sql'] == 0:
            err_sql_count += 1
            continue
        new_sample = sample.copy()
        new_sample['question'], new_sample['nl_input_ids'], new_sample['nl_input_lens'], new_sample['nl_root_idx'], q_datapack = parse_and_process_question(sample['question'], nlp=nlp, tokenizer=codebert_tokenizer)
        assert q_datapack['t_node_count']+1 == len(new_sample['nl_input_lens'])
        new_sample['nl_nt_nodes'] = [v for _,v in q_datapack['nt_nodes'].items()]
        # new_sample['nl_input_tokens'] = [v for _,v in q_datapack['t_nodes'].items()]
        new_sample['nl_input_token'] = q_datapack['nl_input_tokens']
        new_sample['nl_edges'] = q_datapack['edges']
        new_sample['nl_dep_edges'] = q_datapack['dep_edges']
        new_sample['nl_dep_root'] = q_datapack['dep_root']
        assert q_datapack['dep_root'] != -1

        
        sql_datapack, is_valid_sql = parse_and_process_sql(sample['sql'], sample['db_id'], tables_file='preprocess_raw/spider/tables.json')
        
        # sql_datapack, is_valid_sql = parse_and_process_sql(sample['sql'], sample['db_id'], tables_file='preprocess_raw/KaggleDBQA/KaggleDBQA_tables.json')
        if not is_valid_sql:
            err_sql_count += 1
            continue
        new_sample['sql'] = ' '.join(sql_datapack['t_tokens'])
        new_sample['sql_input_tokens'] = sql_datapack['t_input_tokens']
        new_sample['sql_input_ids'] = sql_datapack['t_input_ids']
        new_sample['sql_input_lens'] = sql_datapack['t_input_lens']
        new_sample['sql_nt_nodes'] = [v for _,v in sql_datapack['nt_nodes'].items()]
        new_sample['sql_ast_edges'] = sql_datapack['edges']
        new_sample['sql_t_mask'] = sql_datapack['t_mask']
        new_sample['schema_text'] = ''

        new_sample = add_global_edges(new_sample)
        new_sample = add_sequential_edges(new_sample)
        new_dataset.append(new_sample)
    return new_dataset, err_sql_count

#%%
def preprocess_dataset(name):

    with open(f'{name}.json', 'r') as f:
        dev_json = json.load(f)
    dataset = []
    err_sql_count = 0

    tables_dict = {}
    if 'beam' in name:
        for beam in tqdm(dev_json):
            processed_beam, error_count = preprocess_samples(beam, tables_dict)
            dataset.append(processed_beam)
            err_sql_count += error_count
    else:
        dataset, err_sql_count = preprocess_samples(tqdm(dev_json), tables_dict)
    
    if err_sql_count > 0:
        print('err sql count: ', err_sql_count)
    
    print(f'finished processing dataset <{name}>')
    # sim2 denotes 2nd version of tree simplification (the version in paper).
    with open(f'{name}_sim2.json', 'w') as f:
        json.dump(dataset, f, indent=2)
if __name__ == '__main__':
    
    dataset_names = [
        
        # 'datasets/smbop/ed_smbop_beam_test',
        # 'datasets/smbop/ed_smbop_beam_train',
        # 'datasets/smbop/ed_smbop_beam_dev',
        # 'datasets/natsql/ed_natsql_beam_test',
        # 'datasets/natsql/ed_natsql_beam_spider_tr_train',
        # 'datasets/natsql/ed_natsql_beam_spider_tr_dev',
        # 'datasets/natsql/ed_natsql_beam_train',
        # 'datasets/natsql/ed_natsql_beam_dev',
        # 'datasets/resdnatsql/ed_resdnatsql_beam_test'
        # 'datasets/resdnatsql/ed_resdnatsql_beam_train',
        # 'datasets/resdnatsql/ed_resdnatsql_beam_dev',
    ]

    for name in dataset_names:
        preprocess_dataset(name)

