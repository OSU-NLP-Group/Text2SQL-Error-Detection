def edge_increment(edges:list):
    new_edges = []
    for edge in edges:
        new_edges.append([edge[0]+1, edge[1]+1])
    return new_edges

def get_global_edges(num_nodes:int, offset = 1):
    edges = []
    for i in range(offset, num_nodes+offset):
        edges.append([0, i])
        edges.append([i, 0])
    return edges

def add_global_edges(entry:dict):
    num_nl_t_nodes = len(entry['nl_input_lens'])
    num_nl_nt_nodes = len(entry['nl_nt_nodes'])
    num_sql_nodes = sum(entry['sql_t_mask']) + len(entry['sql_nt_nodes'])
    entry['nl_edges'] = edge_increment(entry['nl_edges'])
    entry['sql_ast_edges'] = edge_increment(entry['sql_ast_edges'])
    entry['nl_dep_global_edges'] = get_global_edges(num_nl_t_nodes)
    entry['nl_consti_global_edges'] = get_global_edges(num_nl_nt_nodes, num_nl_t_nodes+1)
    entry['sql_global_edges'] = get_global_edges(num_sql_nodes)
    return entry

        
        
def _build_seq_edges(starting_idx, num_tokens):
    edges = []
    for i in range(starting_idx, starting_idx + num_tokens-1):
        edges.append([i, i+1])
        edges.append([i+1, i])
    return edges
def add_sequential_edges(entry:dict):
    num_nl_t_nodes = len(entry['nl_input_lens'])
    num_sql_t_nodes = sum(entry['sql_t_mask'])
    entry['nl_seq_edges'] = _build_seq_edges(1, num_nl_t_nodes)
    entry['sql_seq_edges'] = _build_seq_edges(1, num_sql_t_nodes)
    return entry