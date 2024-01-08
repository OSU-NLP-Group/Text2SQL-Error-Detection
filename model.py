import json
import os
import argparse
import logging
import pickle
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import transformers as trf
from preprocessing.dataset_beam import BeamGraphDataset, Indexer
from tqdm import tqdm
from torch_geometric.data import Data, Batch
import numpy as np
import torch_geometric.nn as pyg_nn
from sklearn import metrics
def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='model.py')
    parser.add_argument('--expr_name', type=str)
    parser.add_argument('--train_path', type=str, default='data/train.bin', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev.bin', help='path to dev set (you should not need to modify)')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warmup_factor', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--dir', type=str, default='ED4SP')
    parser.add_argument('--dev_dat', type=str, default='data/ed_spider_dev.dat')
    parser.add_argument('--train_dat', type=str, default='data/ed_spider_train.dat')
    parser.add_argument('--test_dat', type=str, default='data/ed_spider_dev_test.dat')
    parser.add_argument('--gnn_model', type=str, default='GCNConv')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nl_use_dep_edges', action='store_true')
    parser.add_argument('--nl_use_consti_edges', action='store_true')
    parser.add_argument('--use_seq_edges', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--no_sql_graph', action='store_true')
    parser.add_argument('--no_nl_graph', action='store_true')
    
    args = parser.parse_args()
    return args

class BinaryClassificationHead(nn.Module):
    def __init__(self, in_dim, hid_dim, hid_dropout_rate) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(hid_dropout_rate)
        self.proj = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x
class TreeNode:
    def __init__(self, i) -> None:
        self.id = i
        self.children=[]

class BaselineEDModelWithGraph(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.dropout_rate = args.dropout_rate
        self.args = args
        self.CodeBert = trf.RobertaModel.from_pretrained('roberta-base')
        self.nl_use_consti_edges = args.nl_use_consti_edges
        self.nl_use_dep_edges = args.nl_use_dep_edges
        self.use_seq_edges = args.use_seq_edges
        
        self.CODEBERT_DIM = 768

        
        self.NL_GNN_DIM = self.CODEBERT_DIM 
        self.SQL_GNN_DIM = self.CODEBERT_DIM
        self.SQL_NODE_DIM = self.CODEBERT_DIM
        self.NL_NODE_DIM = self.CODEBERT_DIM

            
        if args.gnn_model == 'GCNConv':
            self.use_graph = True
            self.gcn1 = pyg_nn.GCNConv(self.NL_GNN_DIM, self.NL_GNN_DIM)
            self.gcn2 = pyg_nn.GCNConv(self.NL_GNN_DIM, self.NL_GNN_DIM)
            self.gcn3 = pyg_nn.GCNConv(self.NL_GNN_DIM, self.NL_NODE_DIM)
            self.sql_gcn1 = pyg_nn.GCNConv(self.SQL_GNN_DIM, self.SQL_GNN_DIM)
            self.sql_gcn2 = pyg_nn.GCNConv(self.SQL_GNN_DIM, self.SQL_GNN_DIM)
            self.sql_gcn3 = pyg_nn.GCNConv(self.SQL_GNN_DIM, self.SQL_NODE_DIM)
        elif args.gnn_model == 'GATv2':
            self.use_graph = True
            self.gcn1 = pyg_nn.GATv2Conv(self.NL_GNN_DIM, self.NL_GNN_DIM)
            self.gcn2 = pyg_nn.GATv2Conv(self.NL_GNN_DIM, self.NL_GNN_DIM)
            self.gcn3 = pyg_nn.GATv2Conv(self.NL_GNN_DIM, self.NL_NODE_DIM)
            self.sql_gcn1 = pyg_nn.GATv2Conv(self.SQL_GNN_DIM, self.SQL_GNN_DIM)
            self.sql_gcn2 = pyg_nn.GATv2Conv(self.SQL_GNN_DIM, self.SQL_GNN_DIM)
            self.sql_gcn3 = pyg_nn.GATv2Conv(self.SQL_GNN_DIM, self.SQL_NODE_DIM)
        elif args.gnn_model == 'TRF':
            self.use_graph = False
            encoder_layers = nn.TransformerEncoderLayer(self.CODEBERT_DIM, 1, self.CODEBERT_DIM, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 3)
        else:
            self.use_graph = False

        
        if self.use_graph:
            self.ln_dim = self.SQL_NODE_DIM + self.NL_NODE_DIM
        else:
            self.ln_dim = self.CODEBERT_DIM
        
        self.binaryClassifier = BinaryClassificationHead(self.ln_dim, self.ln_dim, self.dropout_rate)
        self.gelu = nn.GELU()
        self.nl_nt_embeddings = nn.Embedding(args.nl_nt_size+1, self.NL_GNN_DIM, padding_idx=0)
        self.sql_nt_embeddings = nn.Embedding(args.sql_nt_size+1, self.SQL_GNN_DIM, padding_idx=0)


    def forward(self, sample, device='cpu'):

        codebert_input = sample['nl_sql'].to(device)
        codebert_input_att_mask = sample['nl_sql_att_mask'].to(device)

        nl_input_mask = sample['nl_mask'].to(device)
        nl_lens = sample['nl_lens'].to(device)
        sql_input_mask = sample['sql_mask'].to(device)
        sql_lens = sample['sql_lens'].to(device)
        sql_t_mask = sample['sql_t_mask'].to(device)

        codebert_output = self.CodeBert(
            codebert_input,
            attention_mask=codebert_input_att_mask
            )[0]

        if not self.use_graph:
            if self.args.gnn_model == 'TRF':
                codebert_output = self.transformer_encoder(codebert_output, src_key_padding_mask=codebert_input_att_mask==0)
            cls1 = codebert_output[:, 0, :]

            clss = cls1 

            o = self.binaryClassifier(clss)
            return o
        
        
        encoded_sql = self.extract_input(codebert_output, sql_lens, sql_input_mask)
        aggregated_sql = self.aggregate_subwords(encoded_sql, sql_lens)

        sql_word_mask = sql_lens>0
        batch_size = sql_t_mask.size()[0]
        if sql_t_mask is not None:
            sql_t_vec = aggregated_sql.masked_select(sql_t_mask.unsqueeze(-1)).reshape([-1, self.CODEBERT_DIM])
        else:
            sql_t_vec = aggregated_sql.masked_select(sql_word_mask.unsqueeze(-1)).reshape([-1, self.CODEBERT_DIM])
        

        encoded_nl = self.extract_input(codebert_output, nl_lens, nl_input_mask)

        aggregated_nl = self.aggregate_subwords(encoded_nl, nl_lens)
        nl_t_mask = nl_lens>0
        nl_t_vec = aggregated_nl.masked_select(nl_t_mask.unsqueeze(-1))
        batch_size, _, feat_dim = codebert_output.size()
        

        if self.nl_use_consti_edges:
            nl_nt_mask = sample['nl_nt_nodes_mask'] > 0
        else:
            nl_nt_mask = None
        sql_nt_mask = sample['sql_nt_nodes_mask']>0
        # ----------------------------
        #####  Prepare Non-terminal Node Vectors #####
        nl_nt_nodes = sample['nl_nt_nodes'].to(device)
        sql_nt_nodes = sample['sql_nt_nodes'].to(device)
        if self.nl_use_consti_edges:        
            nl_nt_nodes = sample['nl_nt_nodes'].to(device)
            nl_nt_embd = self.nl_nt_embeddings(nl_nt_nodes)
            nl_nt_vec = nl_nt_embd.masked_select(nl_nt_mask.unsqueeze(-1)).reshape([-1, self.NL_GNN_DIM])
        else:
            nl_nt_nodes = None
            nl_nt_embd = None
            nl_nt_vec = None
        
        sql_nt_nodes = sample['sql_nt_nodes'].to(device)
        sql_nt_embd = self.sql_nt_embeddings(sql_nt_nodes)
        
        sql_nt_vec = sql_nt_embd.masked_select(sql_nt_mask.unsqueeze(-1)).reshape([-1, self.CODEBERT_DIM])

        # ----------------------------
        
        nl_graph_input_vec, nl_graph_node_counts = self.assemble_graph_node_vec(nl_t_vec, nl_lens>0, nl_nt_vec, nl_nt_mask, batch_size, self.NL_GNN_DIM)
        sql_graph_input_vec, sql_graph_node_counts = self.assemble_graph_node_vec(sql_t_vec, sample['sql_t_node_mask']>0, sql_nt_vec, sql_nt_mask, batch_size, self.SQL_GNN_DIM)
        ##### Done Preparing Non-terminal Node Vectors #####


        nl_graph_data_list = []
        sql_graph_data_list = []
        for i in range(batch_size):
            nl_edges_list = []
            sql_edge_list = [sample['sql_edges'][i]]
            if self.nl_use_dep_edges:
                nl_edges_list.append(sample['nl_dep_edges'][i])
            if self.nl_use_consti_edges:
                nl_edges_list.append(sample['nl_edges'][i])
            if self.use_seq_edges:
                nl_edges_list.append(sample['nl_seq_edges'][i])
                sql_edge_list.append(sample['sql_seq_edges'][i])
            nl_edges = torch.concat(nl_edges_list, dim=-1).to(device)
            sql_edges = torch.concat(sql_edge_list,dim=-1).to(device)
            
            # edges are 1-indexed
            nl_edges -= 1
            sql_edges -= 1
            
            nl_graph_data_list.append(
                Data(
                    x=nl_graph_input_vec[i][:nl_graph_node_counts[i],:],
                    edge_index=nl_edges
                )
            )
            sql_graph_data_list.append(
                Data(
                    x=sql_graph_input_vec[i][:sql_graph_node_counts[i],:],
                    edge_index=sql_edges
                )
            )
        nl_graph_batch = Batch.from_data_list(nl_graph_data_list)
        sql_graph_batch = Batch.from_data_list(sql_graph_data_list)

        
        nl_node_vecs, nl_agg_vecs = self.encode_nl_graphs(nl_graph_batch, nl_graph_node_counts)
        sql_node_vecs, sql_agg_vecs = self.encode_sql_graphs(sql_graph_batch, sql_graph_node_counts)

        agg_nl_vecs = nl_agg_vecs
        agg_sql_vecs = sql_agg_vecs

        agg_graph_vecs = torch.concat([agg_nl_vecs, agg_sql_vecs], dim=-1)
        
        o = self.binaryClassifier(agg_graph_vecs)

        return o
    def _get_tree_from_sample(self, N_t_nodes, edges, node_list):
        
        for edge in edges:
            # We only consider directed edge here
            if min(edge) <= N_t_nodes:
                # edge connecting an terminal node
                if edge[0] > edge[1]:
                    continue
                else:
                    node_list[edge[1]-1].children.append(node_list[edge[0]-1])
            else:
                # edge connecting non-terminal nodes
                if edge[0] > edge[1]:
                    continue
                else:
                    node_list[edge[0]-1].children.append(node_list[edge[1]-1])
            
        return node_list[N_t_nodes]
    
    def encode_nl_graphs(self, batch, graph_node_count):

        if self.args.no_nl_graph:
            encoded_graph = batch.x
        else:
            encoded_graph = self.gelu(self.gcn1(batch.x, batch.edge_index))
            encoded_graph = self.gelu(self.gcn2(batch.x, batch.edge_index))
            encoded_graph = self.gelu(self.gcn3(batch.x, batch.edge_index))
        batch_size = batch.num_graphs
        device = batch.x.device
        dim = encoded_graph.size(-1)
        node_batch_mask = lens2mask(graph_node_count.reshape([1,-1]).squeeze(0)).reshape([batch_size, torch.max(graph_node_count).item()])
        node_vecs = torch.zeros(batch_size, graph_node_count.max().item(), dim).to(device)
        node_vecs = node_vecs.masked_scatter(node_batch_mask.unsqueeze(-1), encoded_graph.reshape([1,-1]))
            
        aggregated_vecs = torch.sum(node_vecs, dim=1)
        aggregated_vecs = aggregated_vecs/graph_node_count.unsqueeze(-1)
        
        return node_vecs, aggregated_vecs

    def encode_sql_graphs(self, batch, graph_node_count):
        if self.args.no_sql_graph:
            encoded_graph = batch.x
        else:
            encoded_graph = self.gelu(self.sql_gcn1(batch.x, batch.edge_index))
            encoded_graph = self.gelu(self.sql_gcn2(batch.x, batch.edge_index))
            encoded_graph = self.gelu(self.sql_gcn3(batch.x, batch.edge_index))
            
        batch_size = batch.num_graphs
        device = batch.x.device
        dim = batch.x.size(-1)
        node_batch_mask = lens2mask(graph_node_count.reshape([1,-1]).squeeze(0)).reshape([batch_size, torch.max(graph_node_count).item()])
        node_vecs = torch.zeros(batch_size, graph_node_count.max().item(), dim).to(device)
        node_vecs = node_vecs.masked_scatter(node_batch_mask.unsqueeze(-1), encoded_graph.reshape([1,-1]))
        aggregated_vecs = torch.sum(node_vecs, dim=1)
        aggregated_vecs = aggregated_vecs/graph_node_count.unsqueeze(-1)
        
        return node_vecs, aggregated_vecs
    
    def assemble_graph_node_vec(self, t_vec, t_mask, nt_vec, nt_mask, batch_size, feat_dim):
        '''
        '''
        device=t_vec.device
        num_t_nodes = torch.sum(t_mask, dim=-1)
        if nt_vec != None:
            num_nt_nodes = torch.sum(nt_mask, dim=-1)
            batch_nt_masks = [torch.tensor([0]*num_t_nodes[i]+[1]*num_nt_nodes[i]).to(device) for i in range(batch_size)]
            nt_mask = torch.nn.utils.rnn.pad_sequence(batch_nt_masks, padding_value=0, batch_first=True)
        else:
            num_nt_nodes = 0
        batch_graph_node_counts = num_t_nodes + num_nt_nodes
        graph_input_len = batch_graph_node_counts.max().item()
        
        t_mask_padding = torch.zeros([batch_size, graph_input_len-t_mask.size(-1)]).to(device)>0
        padded_t_mask = torch.concat((t_mask, t_mask_padding), dim=-1)
        
        
        
        graph_input = torch.zeros([batch_size, graph_input_len, feat_dim]).to(device)
        graph_input = graph_input.masked_scatter(padded_t_mask.unsqueeze(-1), t_vec)
        if nt_vec != None:
            graph_input = graph_input.masked_scatter(nt_mask.unsqueeze(-1)>0, nt_vec)
        return graph_input, batch_graph_node_counts
    
    def extract_input(self, input, lens, input_mask):
        
        device = input.device
        batch_size, input_len, input_dim = input.size()
        
        mask_padding = torch.zeros([batch_size, input_len-input_mask.size()[1]]).to(device)>0
        input_vec_mask = torch.concat([input_mask, mask_padding], dim=-1)>0
        selected_representations = input.masked_select(input_vec_mask.unsqueeze(-1)) 
        subword_mask = lens2mask(lens.reshape([1,-1]).squeeze(0)).reshape([batch_size, -1, torch.max(lens).item()])
        subword_mask = subword_mask.to(device)
        batch_size, max_input_num_subwords, max_subwords_len = subword_mask.size()
        selected_input = torch.zeros(batch_size, max_input_num_subwords, max_subwords_len, input_dim).to(device)
        
        selected_input = selected_input.masked_scatter(subword_mask.unsqueeze(-1), selected_representations)
        return selected_input

    def aggregate_subwords(self, input, subword_lens):
        input = torch.sum(input, dim=-2)
        aggregated_result = input/subword_lens.unsqueeze(-1)
        return aggregated_result
        


def lens2mask(lens):
    '''
    Adapted from LGESQL's implementation.
    '''
    bsize = lens.numel()
    max_len = lens.max()
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks
def my_collate_fn(batch):
    PAD_VALUE = 1
    nl_sql_list = []
    nl_sql_att_mask_list = []
    nl_db_list = []
    nl_db_att_mask_list = []
    nl_sql_db_list = []
    nl_sql_db_att_mask_list = []
    label_list = []
    nl_lens_list = []
    sql_lens_list = []
    nl_mask_list = []
    sql_mask_list = []
    nl_nt_nodes_list = []
    nl_nt_nodes_mask_list = []
    nl_dep_root_list = []
    sql_nt_nodes_list = []
    sql_nt_nodes_mask_list = []
    nl_edges_list = []
    nl_dep_edges_list = []
    sql_edges_list = []
    nl_global_edges_list = []
    nl_consti_global_edges_list = []
    sql_global_edges_list = []
    nl_seq_edges_list = []
    sql_seq_edges_list = []
    sql_t_mask_list = []
    sql_t_node_mask_list = []
    for sample in batch:
        nl_sql_list.append(sample['nl_sql'])
        nl_sql_att_mask_list.append(sample['nl_sql_att_mask'])
        nl_db_list.append(sample['nl_db'])
        nl_db_att_mask_list.append(sample['nl_db_att_mask'])
        nl_sql_db_list.append(sample['nl_sql_db'])
        nl_sql_db_att_mask_list.append(sample['nl_sql_db_att_mask'])
        label_list.append(sample['label'])
        nl_lens_list.append(sample['nl_lens'])
        sql_lens_list.append(sample['sql_lens'])
        nl_mask_list.append(sample['nl_mask'])
        sql_mask_list.append(sample['sql_mask'])
        nl_nt_nodes_list.append(sample['nl_nt_nodes'])
        nl_nt_nodes_mask_list.append(torch.ones(sample['nl_nt_nodes'].size()))
        sql_nt_nodes_list.append(sample['sql_nt_nodes'])
        sql_nt_nodes_mask_list.append(torch.ones(sample['sql_nt_nodes'].size()))
        nl_edges_list.append(sample['nl_edges'])
        nl_dep_edges_list.append(sample['nl_dep_edges'])
        nl_dep_root_list.append(sample['nl_dep_root'])
        sql_edges_list.append(sample['sql_edges'])
        nl_global_edges_list.append(sample['nl_global_edges'])
        nl_consti_global_edges_list.append(sample['nl_consti_global_edges'])
        sql_global_edges_list.append(sample['sql_global_edges'])
        nl_seq_edges_list.append(sample['nl_seq_edges'])
        sql_seq_edges_list.append(sample['sql_seq_edges'])
        if sample['sql_t_mask'] is not None:
            sql_t_mask_list.append(sample['sql_t_mask'])
            sql_t_node_mask_list.append(torch.ones(sum(sample['sql_t_mask'])))
        else:
            t_mask = torch.ones(len(sample['sql_lens']))
            sql_t_mask_list.append(t_mask>0)
            sql_t_node_mask_list.append(torch.ones(t_mask.size())>0)
    return {
        'nl_sql': torch.nn.utils.rnn.pad_sequence(nl_sql_list, batch_first=True, padding_value=PAD_VALUE),
        'nl_sql_att_mask': torch.nn.utils.rnn.pad_sequence(nl_sql_att_mask_list, batch_first=True, padding_value=0),
        'nl_db': torch.nn.utils.rnn.pad_sequence(nl_db_list, batch_first=True, padding_value=PAD_VALUE),
        'nl_db_att_mask': torch.nn.utils.rnn.pad_sequence(nl_db_att_mask_list, batch_first=True, padding_value=0),
        'nl_sql_db': torch.nn.utils.rnn.pad_sequence(nl_sql_db_list, batch_first=True, padding_value=PAD_VALUE),
        'nl_sql_db_att_mask': torch.nn.utils.rnn.pad_sequence(nl_sql_db_att_mask_list, batch_first=True, padding_value=0),
        'label': torch.stack(label_list),
        'nl_lens': torch.nn.utils.rnn.pad_sequence(nl_lens_list, batch_first=True, padding_value=0),
        'sql_lens': torch.nn.utils.rnn.pad_sequence(sql_lens_list, batch_first=True, padding_value=0),
        'nl_mask': torch.nn.utils.rnn.pad_sequence(nl_mask_list, batch_first=True, padding_value=0),
        'sql_mask': torch.nn.utils.rnn.pad_sequence(sql_mask_list, batch_first=True, padding_value=0),
        'sql_t_mask': torch.nn.utils.rnn.pad_sequence(sql_t_mask_list,batch_first=True, padding_value=0),
        'sql_t_node_mask': torch.nn.utils.rnn.pad_sequence(sql_t_node_mask_list,batch_first=True, padding_value=0),
        'nl_nt_nodes': torch.nn.utils.rnn.pad_sequence(nl_nt_nodes_list, batch_first=True, padding_value=0),
        'nl_nt_nodes_mask': torch.nn.utils.rnn.pad_sequence(nl_nt_nodes_mask_list, batch_first=True, padding_value=0),
        'sql_nt_nodes_mask': torch.nn.utils.rnn.pad_sequence(sql_nt_nodes_mask_list, batch_first=True, padding_value=0),
        'sql_nt_nodes': torch.nn.utils.rnn.pad_sequence(sql_nt_nodes_list, batch_first=True, padding_value=0),
        'nl_edges': nl_edges_list,
        'nl_dep_edges': nl_dep_edges_list,
        'sql_edges': sql_edges_list,
        'nl_global_edges': nl_global_edges_list,
        'nl_consti_global_edges': nl_consti_global_edges_list,
        'sql_global_edges': sql_global_edges_list,
        'nl_dep_root': torch.stack(nl_dep_root_list),
        'nl_seq_edges': nl_seq_edges_list,
        'sql_seq_edges': sql_seq_edges_list
    }

def my_collate_fn_beam(batch):
    expanded_samples = []
    beam_indices = [0]
    for beam in batch:
        expanded_samples.extend(beam)
        beam_indices.append(beam_indices[-1]+len(beam))
    flattened_batch = my_collate_fn(expanded_samples)
    flattened_batch['beam_indices'] = torch.tensor(beam_indices)
    return flattened_batch
    


def get_model_roc(eval_json):
    score_list = []
    label_list = []
    for data in eval_json:
        label_list.append(data['label'])
        score_list.append(1-data['score'])
    labels = np.array(label_list)
    scores = np.array(score_list)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = np.round(metrics.auc(fpr,tpr),3)
    return fpr, tpr, auc

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def main():
    args = _parse_args()

    # Reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.gnn_model != 'none':
        assert args.nl_use_consti_edges or args.nl_use_dep_edges or args.gnn_model == 'TRF'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(args.dir, f'experiments/{args.exp_name}/log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print('Experiment: ', args.expr_name)
    logger.info('Experiment: ' + args.expr_name)
    test_name='test'
    print(args.test_dat)
    if 'smbop' in args.test_dat or 'bridge' in args.test_dat or 'natsql' in args.test_dat or 'resdsql' in args.test_dat:
        if 'smbop' in args.test_dat:
            if 'test_gap' in args.test_dat:
                test_name = 'smbop_gap'
            elif 'test_top1' in args.test_dat:
                test_name = 'smbop_top1'
            elif 'beam_test' in args.test_dat:
                test_name = 'smbop_beam'
            elif 'beam_dev' in args.test_dat:
                test_name = 'smbop_dev_beam'
        elif 'bridge' in args.test_dat:
            # test_name = 'bridge_top1'
            if 'beam_test' in args.test_dat:
                test_name = 'bridge_beam'
            elif 'beam_dev' in args.test_dat:
                test_name = 'bridge_dev_beam'
            elif 'kaggle' in args.test_dat:
                test_name = 'bridge_kaggle_beam'
        elif 'resdnatsql' in args.test_dat:
            if 'beam_test' in args.test_dat:
                test_name = 'resdnatsql_beam'
            elif 'beam_dev' in args.test_dat:
                test_name = 'resdnatsql_dev_beam'
        elif 'natsql' in args.test_dat:
            if 'beam_test' in args.test_dat:
                test_name = 'natsql_beam'
            elif 'beam_dev' in args.test_dat:
                test_name = 'natsql_dev_beam'

        else:
            print('ERROR: test set not recognized')
            logger.info('ERROR: test set not recognized')
            exit()
    else:
        print('ERROR: test set not recognized')
        logger.info('ERROR: test set not recognized')
        exit()
    if args.use_beam:
        collate_function = my_collate_fn_beam
    else:
        collate_function = my_collate_fn
        

    if not args.test:
    
        with open(os.path.join(args.dir, args.dev_dat), 'rb') as f:
            dev_set = pickle.load(f)
        with open(os.path.join(args.dir, args.train_dat), 'rb') as f:
            train_set = pickle.load(f)
        
        args.nl_nt_size = train_set.nl_nt_indexer.get_vocab_size()
        args.sql_nt_size = train_set.sql_nt_indexer.get_vocab_size()

        train_dataloader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True, collate_fn=collate_function, worker_init_fn=seed_worker, generator=g)
        dev_dataloader = DataLoader(dev_set, batch_size = 1, collate_fn=collate_function, worker_init_fn=seed_worker, generator=g)
        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        model = BaselineEDModelWithGraph(args)
        model = nn.DataParallel(model)
        model = model.to(device)
        
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr
        )
        if args.warmup_factor > 0:
            lr_scheduler = trf.get_scheduler(
                name='linear',
                optimizer=optim, 
                num_warmup_steps=len(train_dataloader)*args.epochs//args.warmup_factor,
                num_training_steps=len(train_dataloader)*args.epochs)
        
        best_auc = 0
        best_dev_acc = 0
        best_acc_epoch = 0
        best_auc_epoch = 0
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(args.epochs):
            model.train()
            interval_loss_list = []
            for step, dat in enumerate(train_dataloader):
                label = dat['label'].to(device)
                
                logit = model(dat, device=device)

                ed_loss = loss_fn(logit, label.float())
                

                loss = ed_loss

                interval_loss_list.append(loss.item())
                if (step+1) % 50 == 0:
                    print(f'{step+1}/{len(train_dataloader)} - avg loss: {np.mean(interval_loss_list)}')
                    logger.info(f'{step+1}/{len(train_dataloader)} - avg loss: {np.mean(interval_loss_list)}')
                    interval_loss_list = []
                    

                loss.backward()
                optim.step()
                if args.warmup_factor > 0:
                    lr_scheduler.step()
                optim.zero_grad()
            
            model.eval()
            correct = 0
            test_results = []
            threshold = 0.5
            N_samples = 0
            for idx, dat in enumerate(dev_dataloader):
                label = dat['label'].to(device)
                with torch.no_grad():
                    logit = model(dat, device=device)
                score = torch.sigmoid(logit)

                eval_result = torch.eq(score>threshold, label > 0)
                correct += eval_result.sum()

                effective_batch_size = dat['label'].size()[0]
                N_samples += effective_batch_size
                for i in range(effective_batch_size):
                    test_results.append({
                    'id': idx,
                    'pred': 1 if score[i].item() > threshold else 0,
                    'label': label[i].item(),
                    'score': score[i].item()
                    })
            
            _, _, auc = get_model_roc(test_results)
            if auc > best_auc:
                best_auc = auc
                best_auc_epoch = epoch
            if correct > best_dev_acc:
                best_dev_acc = correct
                best_acc_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss':loss
                }, os.path.join(args.dir, f'experiments/{args.exp_name}/checkpoint.pth'))
                
            print(f'epoch {epoch+1} dev acc: {correct}/{N_samples} =  = {correct/N_samples} - auc: {auc}')
            
            logger.info(f'epoch {epoch+1} dev acc: {correct}/{N_samples} = {correct/N_samples} - auc: {auc}')

            print(f'[best acc] epoch {best_acc_epoch+1} dev acc: {best_dev_acc}/{N_samples} = {best_dev_acc/N_samples}')
            print(f'[best auc] epoch {best_auc_epoch+1} dev auc: {best_auc}')


            logger.info(f'[best acc] epoch {best_acc_epoch+1} dev acc: {best_dev_acc}/{N_samples} = {best_dev_acc/N_samples}')
            logger.info(f'[best auc] epoch {best_auc_epoch+1} dev auc: {best_auc}')
    # Inference
    with open(os.path.join(args.dir, args.test_dat), 'rb') as f:
        test_set = pickle.load(f)

    test_dataloader = DataLoader(test_set, batch_size = 1, collate_fn=collate_function, worker_init_fn=seed_worker, generator=g)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    args.nl_nt_size = test_set.nl_nt_indexer.get_vocab_size()
    args.sql_nt_size = test_set.sql_nt_indexer.get_vocab_size()
        
    model = nn.DataParallel(BaselineEDModelWithGraph(args))
    model.load_state_dict(torch.load(f'{args.dir}/experiments/{args.exp_name}/checkpoint.pth', map_location=torch.device('cpu'))['model_state_dict'])
    model = model.to(device)
    model.eval()
    correct = 0
    test_results = []
    N_samples = 0
    threshold=0.5
    top1_correct = 0
    top1_test_results = []
    flat_test_results = []
    for idx, dat in enumerate(tqdm(test_dataloader)):
        label = dat['label'].to(device)
        with torch.no_grad():
            logit = model(dat, device=device)
        score = torch.sigmoid(logit)
        
        eval_result = torch.eq(score>threshold, label > 0)
        correct += eval_result.sum()
                
        effective_batch_size = dat['label'].size()[0]
        N_samples += effective_batch_size
        batch_result = []
        for i in range(effective_batch_size):
            eval_result = {
            'id': idx,
            'pred': 1 if score[i].item() > threshold else 0,
            'label': label[i].item(),
            'score': score[i].item()
            }

            
            if i == 0:
                top1_correct += 1 if (label[i].item()>0) == (score[i].item()>threshold) else 0
                top1_test_results.append(eval_result)
            batch_result.append(eval_result)
        test_results.append(batch_result)
        flat_test_results.extend(batch_result)

    with open(os.path.join(args.dir, f'experiments/{args.exp_name}/eval_{test_name}.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    _, _, auc = get_model_roc(flat_test_results)
    _, _, top1_auc = get_model_roc(top1_test_results)
    print(f'test acc: {correct}/{N_samples} = {correct/N_samples}')
    print(f'test auc: {auc}')
    logger.info(f'test acc: {correct}/{N_samples} = {correct/N_samples}')
    logger.info(f'test auc: {auc}')
    print(f'test top1 acc: {top1_correct}/{len(test_dataloader)} = {top1_correct/len(test_dataloader)}')
    print(f'test top1 auc: {top1_auc}')
    logger.info(f'test top1 acc: {top1_correct}/{len(test_dataloader)} = {top1_correct/len(test_dataloader)}')
    logger.info(f'test top1 auc: {top1_auc}')


if __name__ == '__main__':
    main()
    