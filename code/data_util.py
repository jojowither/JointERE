import torch
import torch.utils.data as Data

import numpy as np
import copy
import json
import pickle

class Schema(dict):
    '''
    Parse and Represent Schema of Knowledge, and provide conversion between 
    tags representations and entity/relation indexes.
    '''
    
    UNKOWN_TAG = "<UNKNOWN>"
    PAD_TAG = "<PAD>"
    REL_PAD = 'Rel-Pad'
    REL_NONE = 'Rel-None'
    
    def __init__(self, schema_data):
        '''
        Parse schema dictionary file and construct TagDict for tag/index 
        conversion, as well as type lookup from index of tag.
        Input:
            schema_data:
                The path to schema dictionary file
        '''
        
        raw_dict = "".join(readfile(schema_data))
        dict2json = "".join(raw_dict.split()[2:])

        json_acceptable_string = dict2json.replace("'", "\"")
        super().__init__(json.loads(json_acceptable_string))
        
        self.ent2ix = EntTagDict(self)
        self.rel2ix = RelTagDict(self)
        
        self.rid2tag = self.convert_rid()
        
    def eid_from_tag_ix(self, ix):
        '''
        Provide lookup of entity id from index of tag
        Input:
            ix:
                An index of entity tag
        Output:
            entity id with respect to ix
        '''
        
        return self.ent2ix.ix_base_type(ix)
    
    def rid_from_tag_ix(self, ix):
        '''
        Provide lookup of relation id from index of tag
        Input:
            ix:
                An index of relation tag
        Output:
            relation id with respect to ix
        '''
            
        return self.rel2ix.ix_base_type(ix)
    
    def convert_rid(self):
        convert_rid_dict = {}
        for k,v in self['relation'].items():
            convert_rid_dict[v['rid']] = v['tag']

        return convert_rid_dict



class TagDict(dict):
    '''
    Base tag-index dictionary data structure
    It also stores a list to provide index-tag lookup.
    '''

    def __init__(self, schema):
        self.tags = self.define(schema)
        super().__init__(((t, i) for i, t in enumerate(self.tags)))

    def inv(self, idx):
        return self.tags[idx]

    def define(self, schema):
        raise NotImplementedError()

    def ix_base_type(self, idx):
        raise NotImplementedError()

class EntTagDict(TagDict):

    def define(self, schema):
        '''
        Define entity tags in presumed BIO scheme.
        Input:
            schema:
                An instance of data_util.Schema
        Output:
            bio_tags:
                A list of tags of Begining/Intermediate of entity and non-entity
        '''

        tag_type = list(schema['tagging'])
        entity_tag = [ent['tag'] for ent in schema['entity'].values()]

        bio_tags = []
        for t in tag_type:
            for e in entity_tag:
                if t != 'O':
                    bio_tags.append(t + '-' + e)

        bio_tags = [schema.UNKOWN_TAG, schema.PAD_TAG] + bio_tags + ['O']

        return bio_tags

    def ix_base_type(self, idx):
        '''
        Infer base entity type from index of entity BIO tag as the construction
        sequence in define().
        '''
        
        return int(idx - 1) % (len(self) // 2 - 1)

class RelTagDict(TagDict):

    def define(self, schema):
        relation_tags = [rel['tag'] for rel in schema['relation'].values()]

        relation_tags = [schema.REL_PAD, schema.REL_NONE]  + relation_tags
        return relation_tags

    def ix_base_type(self, idx):
        return int(idx)
    



# ====================================================

class BIOLoader(Data.DataLoader):
    
    def __init__(self, data_path, max_len, batch_size, schema, rel_be_filtered=None, 
                 word_to_ix=None, shuffle=False, device=torch.device('cpu')):
        
        '''
        Load corpus and dictionary if available to initiate a torch DataLoader
        Input:
            data_path:
                The string of path to BIO-format corpus.
            max_len:
                The maximal tokens allowed in a sentence.
            batch_size:
                The batch_size parameter as a torch DataLoader.
            schema:
                An instance of data_util.Schema
            rel_be_filtered:
                Select the relation you want to filter out in the experiment.
            word_to_ix: optional
                The word (token) dictionary to index. Use the dict to map sentences
                into indexed sequences if provided, or try to load it from disk if 
                the path is provided.
                If the dictionary does not present in the path, this class would
                write the newly parsed dictionary to the path.
            shuffle: optional
                The shuffle parameter as a torch Dataloader.
            device: optional
                The device at which the dataset is going to be loaded.
        '''
        
        self.max_len = max_len
    
        w2ix_path = None
        if isinstance(word_to_ix, str):
            w2ix_path, word_to_ix = word_to_ix, None
            with open(w2ix_path, 'rb') as fin:
                word_to_ix = pickle.load(fin)
                
                
        self.word_to_ix, self.vocab_size, self.raw_input, *results = self.preprocess(data_path, 
                                                                      word_to_ix, schema, rel_be_filtered)
        
        if w2ix_path is not None and word_to_ix is None:
            with open(w2ix_path, 'wb') as fout:
                pickle.dump(self.word_to_ix, fout)
                                                                     
        torch_dataset = Data.TensorDataset(*(x.to(device) for x in results))
        
        super().__init__(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False
        )
    
    
    
    '''
    function split_to_list()
    arg rel_be_filtered need not be filled if not in experiment
    '''
    def preprocess(self, data, word_to_ix, schema, rel_be_filtered=None):
        content = readfile(data)
        sent_list, ent_list, rel_list = split_to_list(content, schema, rel_be_filtered)
        word_to_ix = word_to_ix or word2index(sent_list)
        reserved_index = filter_len(sent_list, self.max_len)
        filter_word, filter_ent, filter_rel = filter_sentence(reserved_index, sent_list, ent_list, rel_list)
        f_w, f_e, f_r = deep_copy_lists(filter_word, filter_ent, filter_rel)
        input_padded, ent_padded, rel_padded = pad_all(f_w, f_e, f_r, self.max_len)
        #================================================
        input_var = prepare_all(input_padded, word_to_ix)
        ent_var = prepare_all(ent_padded, schema.ent2ix)
        rel_var = prepare_rel(rel_padded, schema.rel2ix, self.max_len)
        #================================================


        reserved_index = torch.from_numpy(np.asarray(reserved_index))
        
        
        return word_to_ix, len(word_to_ix), sent_list, input_var, ent_var, rel_var, reserved_index




# ==================================================

def readfile(data):
    with open(data, "r", encoding="utf-8") as f:
        content = f.read().splitlines()
        
    return content



def filter_relation(word_set, rel_be_filtered):
    
    reserved_rel = []
    for w_set in word_set:
        if all(r not in w_set for r in rel_be_filtered):
            reserved_rel.append(w_set)
            
    if not reserved_rel:
        return Schema.REL_NONE
    else:
        return reserved_rel


    
def filter_entity(word_set, rel_be_filtered, schema):
    reserved_rel_type = schema.rel2ix.keys() - rel_be_filtered - {schema.REL_PAD,schema.REL_NONE}
    reserved_tag = []
    
    for k,v in schema['relation'].items():
        for single_r in reserved_rel_type:
            if schema['relation'][k]['tag']==single_r:
                reserved_tag.append(k)   
                
    reserved_arguments = []
    for t in reserved_tag:
        reserved_arguments.extend(list(schema['relation'][t]['arguments'].values()))
        
    reserved_arguments = list(set(reserved_arguments))
    
    reserved_ent_tag = []
    for arg in reserved_arguments:
        reserved_ent_tag.append(schema['entity'][arg]['tag'])
    
    
    try:
        testerror = word_set.split('-')[1]
    except IndexError:
        return 'O'
    else:
        if word_set.split('-')[1] in reserved_ent_tag:
            return word_set
        else:
            return 'O'
        
    


def get_word_and_label(_content, start_w, end_w, rel_be_filtered, schema):
    word_list = []
    ent_list = []
    rel_list = []
    
    for word_set in _content[start_w:end_w]:
        word_set = word_set.split()
        if len(word_set)==1:
            word_list.append(' ')
            ent_list.append('O')
            rel_list.append(Schema.REL_NONE)
        
        else:
            word_list.append(word_set[0])

            
            
            if rel_be_filtered:
                reserved_ent = filter_entity(word_set[1], rel_be_filtered, schema)
                ent_list.append(reserved_ent)      
            else:
                ent_list.append(word_set[1])
            
            

            try:
                testerror = word_set[2]
            except IndexError:
                rel_list.append(Schema.REL_NONE)
            else:
                if rel_be_filtered:
                    reserved_rel = filter_relation(word_set[2:], rel_be_filtered)
                    rel_list.append(reserved_rel)
                    
                else:
                    rel_list.append(word_set[2:])
                
    
    return word_list, ent_list, rel_list

def split_to_list(content, schema, rel_be_filtered=None):
    init = 0
    word_list = []
    ent_list = []
    rel_list = []

    for now_token, c in enumerate(content):
        if c == '':
            words, ents, rels = get_word_and_label(content, init, now_token, rel_be_filtered, schema)
            init = now_token + 1
            word_list.append(words)
            ent_list.append(ents)
            rel_list.append(rels)
            
    return word_list, ent_list, rel_list

# ==================================================

def word2index(sent_list):
    vocab = { Schema.UNKOWN_TAG, Schema.PAD_TAG }
    vocab.update((word for sent in sent_list for word in sent))
    
    return { w: i for i, w in enumerate(vocab) }

def dict_inverse(tag_to_ix):
    return {v: k for k, v in tag_to_ix.items()}
    

def index2tag(indexs, ix_to):
    return [ix_to[i] for i in indexs.cpu().numpy()]


# ==================================================

def find_max_len(word_list):
    max_len = 0
    for i in range(len(word_list)):
        if max_len < len(word_list[i]):
            max_len = len(word_list[i])
            
    return max_len

# ====== filter the length of sentence more than MAX_LEN =======

def filter_len(word_list, max_len):
    reserved_index = []
    for i in range(len(word_list)):
        if len(word_list[i]) < max_len:
            reserved_index.append(i)
            
    return reserved_index


def filter_sentence(reserved_index, word_list, ent_list, rel_list):
    filter_word = list(word_list[i] for i in reserved_index)
    filter_ent = list(ent_list[i] for i in reserved_index)
    filter_rel = list(rel_list[i] for i in reserved_index)
    return filter_word, filter_ent, filter_rel

# ==================================================

def pad_seq(seq, pad, max_len):
    seq += [pad for i in range(max_len - len(seq))]
    return seq

def pad_all(filter_word, filter_ent, filter_rel, max_len):
    input_padded = [pad_seq(s, Schema.PAD_TAG, max_len) for s in filter_word]
    ent_padded = [pad_seq(s, Schema.PAD_TAG, max_len) for s in filter_ent]
    
    # rel-none
    rel_padded = [pad_seq(s, Schema.REL_NONE, max_len) for s in filter_rel]
    
    return input_padded, ent_padded, rel_padded

def deep_copy_lists(filter_word, filter_ent, filter_rel):
    f_w = copy.deepcopy(filter_word)
    f_e = copy.deepcopy(filter_ent)
    f_r = copy.deepcopy(filter_rel)
    
    return f_w, f_e, f_r

# ==================================================

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        try:
            idxs.append(to_ix[w])            
        except KeyError:
            idxs.append(to_ix[Schema.UNKOWN_TAG])
    
    return torch.tensor(idxs, dtype=torch.long)

def prepare_all(seqs, to_ix):
    seq_list = []
    for i in range(len(seqs)):
        seq_list.append(prepare_sequence(seqs[i], to_ix))
        
    seq_list = torch.stack(seq_list)
        
    return seq_list



def prepare_rel(rel_padded, to_ix, max_len):
    '''
    Prepare relation label data structure
    Output:
        rel_ptr: BATCH*LEN*LEN
            Labels for whether a relation exists from the former to the later token
    '''
    
    rel_ptr = torch.zeros(len(rel_padded), max_len, max_len, dtype=torch.long) 
    
    # 對當前的token，去比較之前所有出現過的entity，是否有關係，建成矩陣
    # [B*ML*ML]，第二維ML是當前token，第三維ML是根據當前token對之前出現過的entity紀錄關係，以index紀錄
    for i, rel_seq in enumerate(rel_padded):
        rel_dict = {}
        for j, token_seq in enumerate(rel_seq):
            rel_ptr[i][j][:j+1] = to_ix[Schema.REL_NONE]
            if token_seq != Schema.REL_NONE:
                for k, rel in enumerate(token_seq):

                    # if 是第一次出現，紀錄後面數字(標第幾對)和關係位置(A OR B)
                    # 假如下次出現又是同個關係位置(A)，依然紀錄
                    # 直到下次出現關係位置B，依照之前紀錄的A位置的字，然後在第三維去標關係

                    rel_token = rel.split('-')
                    if rel_token[1] not in rel_dict:
                        rel_dict[rel_token[1]] = {'rel':rel_token[0], 'loc':rel_token[2], 'idx':[j]}

                    elif rel_token[1] in rel_dict and rel_dict[rel_token[1]]['loc']==rel_token[2]:
                        rel_dict[rel_token[1]]['idx'].append(j)

                    else:
                        record_loc = rel_dict[rel_token[1]]['idx']
                        for idxx in record_loc:
                            rel_ptr[i][j][idxx] = to_ix[rel_token[0]]
                            
    return rel_ptr

