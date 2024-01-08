from transformers import RobertaTokenizer, AutoTokenizer
import json
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

MAX_SEQ_LEN = 512

class BeamGraphDataset(Dataset):
    def __init__(self, data_dir = '', is_train=False, indexer_name='') -> None:
        super().__init__()
        self.is_train = is_train
        if is_train:
            self.sql_nt_indexer = Indexer()
            self.nl_nt_indexer = Indexer()
        else:
            if indexer_name == '':
                indexer_name=data_dir.split('.')[-2].replace('dev', 'train').replace('test', 'train')

            with open(f'{indexer_name}_indexer_sql_nt.pkl'.replace('_gg', ''), 'rb') as f:
                self.sql_nt_indexer = pickle.load(f)

                print('Loaded sql nt indexer with vocab size: ', self.sql_nt_indexer.get_vocab_size())
            with open(f'{indexer_name}_indexer_nl_nt.pkl'.replace('_gg', ''), 'rb') as f:    
                self.nl_nt_indexer = pickle.load(f)
                print('Loaded nl nt indexer with vocab size: ', self.nl_nt_indexer.get_vocab_size())
        self.samples = self._load_data(data_dir)    

        self.size = len(self.samples)
    def __len__(self):
        return self.size
    def _get_sample(self, sample):
        return {
            'nl_sql': sample['nl_sql'],
            'nl_sql_att_mask': sample['nl_sql_att_mask'],
            'nl_db': sample['nl_db'],
            'nl_db_att_mask': sample['nl_db_att_mask'],
            'nl_sql_db': sample['nl_sql_db'],
            'nl_sql_db_att_mask': sample['nl_sql_db_att_mask'],
            'label': sample['label'],
            'nl_lens': sample['nl_lens'],
            'sql_lens': sample['sql_lens'],
            'nl_mask': sample['nl_mask'],
            'sql_mask': sample['sql_mask'],
            'sql_t_mask': sample['sql_t_mask'] if 'sql_t_mask' in sample else None,
            'nl_nt_nodes': sample['nl_nt_nodes'],
            'sql_nt_nodes': sample['sql_nt_nodes'],
            'nl_edges': sample['nl_edges'],
            'nl_dep_edges': sample['nl_dep_edges'],
            'nl_dep_root': sample['nl_dep_root'],
            'sql_edges': sample['sql_edges'],
            'nl_global_edges': sample['nl_global_edges'],
            'nl_consti_global_edges': sample['nl_consti_global_edges'],
            'sql_global_edges': sample['sql_global_edges'],
            'nl_seq_edges': sample['nl_seq_edges'],
            'sql_seq_edges': sample['sql_seq_edges'],
        }
    def __getitem__(self, idx):
        return [self._get_sample(sample) for sample in self.samples[idx]]

    def _load_data(self, data_dir):
        samples = []
        codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        grappa_tokenizer = AutoTokenizer.from_pretrained('Salesforce/grappa_large_jnt')
        with open(data_dir, 'r') as fp:
            data_json = json.load(fp)
        
        for beam in tqdm(data_json):
            if len(beam) == 0:
                print('empty beam, skipped')
                continue
            beam_samples = []
            for sample in beam:
                cls_id = codebert_tokenizer.convert_tokens_to_ids([codebert_tokenizer.cls_token])
                sep_id = codebert_tokenizer.convert_tokens_to_ids([codebert_tokenizer.sep_token])
                eos_id = codebert_tokenizer.convert_tokens_to_ids([codebert_tokenizer.eos_token])
                
                schema_tokens = codebert_tokenizer.tokenize(sample['schema_text'].replace('</s>', codebert_tokenizer.sep_token))
                cb_schema_ids = codebert_tokenizer.convert_tokens_to_ids(schema_tokens)
                grappa_schema_ids = grappa_tokenizer.convert_tokens_to_ids(grappa_tokenizer.tokenize(sample['schema_text']))
                
                nl_sql_input_ids = cls_id + sample['nl_input_ids'] + sep_id + sample['sql_input_ids'] + eos_id
                nl_sql_att_mask = torch.ones([len(nl_sql_input_ids)])
                nl_sql_input = torch.LongTensor(nl_sql_input_ids)
                nl_sql_db_input_ids = cls_id + sample['nl_input_ids'] + sep_id + sample['sql_input_ids'] + sep_id + cb_schema_ids
                nl_sql_db_input_ids = nl_sql_db_input_ids[:-1][:MAX_SEQ_LEN-1] + sep_id
                nl_sql_db_att_mask = torch.ones([len(nl_sql_db_input_ids)])
                nl_sql_db_input = torch.LongTensor(nl_sql_db_input_ids)
                nl_db_input_ids = cls_id + sample['nl_input_ids'] + sep_id + grappa_schema_ids
                nl_db_input_ids = nl_sql_db_input_ids[:-1][:MAX_SEQ_LEN-1] + sep_id
                nl_db_att_mask = torch.ones([len(nl_db_input_ids)])
                nl_db_input = torch.LongTensor(nl_db_input_ids)
                nl_lens = torch.LongTensor(sample['nl_input_lens'])
                nl_mask = torch.LongTensor([0] + [1]*sum(sample['nl_input_lens'])) # add [CLS]
                sql_lens = torch.LongTensor(sample['sql_input_lens'])
                sql_mask = torch.LongTensor([0]*(2+sum(sample['nl_input_lens'])) + [1]*len(sample['sql_input_ids'])) # add [CLS] and [SEP]
                sql_t_mask = torch.LongTensor(sample['sql_t_mask'])>0

                assert sql_mask.size()[0] == nl_sql_input.size()[0] - 1
                if self.is_train:
                    nl_nt_nodes = torch.LongTensor([self.nl_nt_indexer.index_and_add(node) for node in sample['nl_nt_nodes']])
                    sql_nt_nodes = torch.LongTensor([self.sql_nt_indexer.index_and_add(node) for node in sample['sql_nt_nodes']])
                else:
                    nl_nt_nodes = torch.LongTensor([self.nl_nt_indexer.index(node) for node in sample['nl_nt_nodes']])
                    sql_nt_nodes = torch.LongTensor([self.sql_nt_indexer.index(node) for node in sample['sql_nt_nodes']])
                
                nl_edges = torch.LongTensor(sample['nl_edges']).t().contiguous()
                nl_dep_edges = torch.LongTensor(sample['nl_dep_edges']).t().contiguous() # stanza id starts from 1
                sql_edges = torch.LongTensor(sample['sql_ast_edges']).t().contiguous()            
                nl_dep_root = torch.LongTensor([sample['nl_dep_root']])
                nl_global_edges = torch.LongTensor(sample['nl_dep_global_edges']).t().contiguous()
                nl_consti_global_edges = torch.LongTensor(sample['nl_consti_global_edges']).t().contiguous()
                sql_global_edges = torch.LongTensor(sample['sql_global_edges']).t().contiguous()
                nl_seq_edges = torch.LongTensor(sample['nl_seq_edges']).t().contiguous()
                sql_seq_edges = torch.LongTensor(sample['sql_seq_edges']).t().contiguous()
                # check the number of nodes agrees with edge indexes
                assert nl_edges.max() == len(sample['nl_input_lens']) + len(sample['nl_nt_nodes'])
                assert sql_edges.max() == sum(sample['sql_t_mask']) + len(sample['sql_nt_nodes'])
                beam_samples.append({
                    'nl_sql': nl_sql_input,
                    'nl_sql_att_mask': nl_sql_att_mask,
                    'nl_db': nl_db_input,
                    'nl_db_att_mask':nl_db_att_mask,
                    'nl_sql_db': nl_sql_db_input,
                    'nl_sql_db_att_mask':nl_sql_db_att_mask,
                    'label': torch.LongTensor([sample['label']]),
                    'nl_lens': nl_lens,
                    'sql_lens': sql_lens,
                    'nl_mask': nl_mask,
                    'sql_mask': sql_mask,
                    'sql_t_mask': sql_t_mask,
                    'nl_nt_nodes': nl_nt_nodes,
                    'sql_nt_nodes': sql_nt_nodes,
                    'nl_edges': nl_edges,
                    'nl_dep_edges': nl_dep_edges,
                    'nl_dep_root': nl_dep_root,
                    'sql_edges': sql_edges,
                    'nl_global_edges': nl_global_edges,
                    'nl_consti_global_edges': nl_consti_global_edges,
                    'sql_global_edges': sql_global_edges,
                    'nl_seq_edges': nl_seq_edges,
                    'sql_seq_edges': sql_seq_edges
                })
            samples.append(beam_samples)
        return samples
class Indexer:
    def __init__(self) -> None:
        self.objs_to_id = {}
        self.id_to_objs = {}
        self.vocab_size = 0
    
    def index_and_add(self, token):
        if token in self.objs_to_id:
            return self.objs_to_id[token]
        else:
            self.vocab_size += 1
            new_id = self.vocab_size
            self.objs_to_id[token] = new_id
            self.id_to_objs[new_id] = token
            
            return new_id
    
    def index(self, token):
        if token in self.objs_to_id:
            return self.objs_to_id[token]
        else:
            return 0
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def decode(self, id):
        if id in self.id_to_objs:
            return self.id_to_objs[id]
        else:
            return ''
    
if __name__ == '__main__':
    
    dataset_names = [
        
        # 'datasets/smbop/ed_smbop_beam_train_sim2',
        # 'datasets/smbop/ed_smbop_beam_dev_sim2',
        'datasets/smbop/ed_smbop_beam_test_sim2',

        # 'datasets/natsql/ed_natsql_beam_train_sim2',
        # 'datasets/natsql/ed_natsql_beam_dev_sim2',
        # 'datasets/natsql/ed_natsql_beam_test_sim2',
        

        # 'datasets/natsql/ed_natsql_beam_spider_tr_train_sim2',
        # 'datasets/natsql/ed_natsql_beam_spider_tr_dev_sim2',
        
        # 'datasets/natsql/ed_natsql_beam_train_ori',
        # 'datasets/natsql/ed_natsql_beam_dev_ori',
        # 'datasets/natsql/ed_natsql_beam_test_ori',
        
        # 'datasets/resdnatsql/ed_resdnatsql_beam_train_sim2',
        # 'datasets/resdnatsql/ed_resdnatsql_beam_dev_sim2',
        # 'datasets/resdnatsql/ed_resdnatsql_beam_test_sim2',
        
    ]
    ### For processing training data, both are ''.
    indexer_name, suffix='', ''


    # For processing evaluation data, choose the correct indexer (same as training setting). The suffix is used for distinguishing different indexers in the output .dat file.
    # Suffix = '' for evaluations on the same parser as training.
    #
    # For cross-parser evaluations, suffix is the abbreviation of source parser.
    #     smbp -> SmBoP
    #     ntsq -> NatSQL
    #     rsdntsq -> ResdNatSQL
    # The following commented lines are example usages.

    indexer_name, suffix = 'datasets/smbop/ed_smbop_beam_train_sim2', '_smbp' # for cross-parser evaluation
    # indexer_name, suffix = 'datasets/smbop/ed_smbop_beam_train_sim2', '' # for evaluation on SmBoP
    # indexer_name, suffix = 'datasets/natsql/ed_natsql_beam_train_sim2','_ntsq'
    # indexer_name, suffix = 'datasets/natsql/ed_natsql_beam_train_sim2',''
    # indexer_name, suffix = 'datasets/resdnatsql/ed_resdnatsql_beam_train_sim2','_rsdntsq'
    # indexer_name, suffix = 'datasets/resdnatsql/ed_resdnatsql_beam_train_sim2',''
    

    # For ablation studies 
    # _ori: Without graph simplification.

    # indexer_name, suffix = 'datasets/natsql/ed_natsql_beam_train_ori', 'ntsq'

    ### ntsq_spider_tr: Data from NatSQL collected by in-domain errors.
    
    # indexer_name, suffix='datasets/natsql/ed_natsql_beam_spider_tr_train_sim2', '_ntsq_spider_tr'    
    
    suffix = ''
    for name in dataset_names:
        is_train = 'train' in name
        dataset = BeamGraphDataset(f'{name}.json', is_train=is_train, indexer_name=indexer_name)
        if is_train:
            with open(f'{name}_indexer_sql_nt.pkl', 'wb') as f:
                print('Saved indexer for sql non-terminal nodes, with vocab size: ',  dataset.sql_nt_indexer.get_vocab_size())
                pickle.dump(dataset.sql_nt_indexer,f)
            with open(f'{name}_indexer_nl_nt.pkl', 'wb') as f:
                pickle.dump(dataset.nl_nt_indexer,f)
                print('Saved indexer for nl non-terminal nodes, with vocab size: ',  dataset.nl_nt_indexer.get_vocab_size())

        with open(f'{name}{suffix}.dat', 'wb') as f:
            pickle.dump(dataset, f)


