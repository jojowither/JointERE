import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
import warnings
import matplotlib.pyplot as plt


def evaluate_data(model, data_loader, schema, isTrueEnt=False, silent=False, rel_detail=False, analyze=False):
    
    y_ent_true_all, y_ent_pred_all = [], []
    y_rel_true_all, y_rel_pred_all = [], []
    tps, fps, tns, fns = 0, 0, 0, 0
    total_r_error = 0
    
    anay_rel_true = {}
    anay_rel_pred = {}
    anay_rel_pospred = {}
    for i in range(len(schema['relation'])):
        anay_rel_true[i]=[]
        anay_rel_pred[i]=[]
        anay_rel_pospred[i]=[]
    
    
    if silent:
        warnings.simplefilter('ignore')
    else:
        warnings.filterwarnings('always')
        
        
    with torch.no_grad():
        for batch_x, batch_ent, batch_rel, batch_index in data_loader:
            model.eval()
            
            if isTrueEnt:
                ent_output, rel_output = model(batch_x, batch_ent)
            else:
                ent_output, rel_output = model(batch_x)
            
            batchsize, max_len = batch_ent.size()
            

            anay_true, anay_pred, anay_pospred, r_err_count, *(score_num) = batch_decode(ent_output.cpu(), rel_output.cpu(), 
                                                                           batch_index,  data_loader.raw_input, 
                                                                           batch_ent.cpu(), batch_rel.cpu(), 
                                                                           schema, silent=silent, analyze=analyze)
            
            y_true_ent, y_pred_ent, y_true_rel, y_pred_rel, tp, fp, tn, fn = score_num
            
            y_ent_true_all.extend(y_true_ent)
            y_ent_pred_all.extend(y_pred_ent)
            y_rel_true_all.extend(y_true_rel)
            y_rel_pred_all.extend(y_pred_rel)
            
            
            tps += tp
            fps += fp
            tns += tn
            fns += fn           
            total_r_error += r_err_count
            
            if analyze:
                for r in anay_true:
                    anay_rel_true[r].extend(anay_true[r])
                for r in anay_pred:
                    anay_rel_pred[r].extend(anay_pred[r])
                for r in anay_pospred:
                    anay_rel_pospred[r].extend(anay_pospred[r])
                    
            
            
            
    e_score = precision_recall_fscore_support(y_ent_true_all, y_ent_pred_all, average='micro', 
                                               labels=range(len(schema['entity'])))[:-1]
    
    er_score = precision_recall_fscore_support(y_rel_true_all, y_rel_pred_all, average='micro', 
                                               labels=range(len(schema['relation'])))[:-1]
    
    
    print()
    print("Entity detection score")
    print("precision  \t recall  \t fbeta_score")
    print("{:.3f} \t\t {:.3f} \t\t {:.3f} \t".format(*e_score))
    
    print("Entity+Relation detection score ")
    print("precision  \t recall  \t fbeta_score  \t")
    print("{:.3} \t\t {:.3} \t\t {:.3} \t".format(*er_score))
    
    print('confusion matrix ')
    print('TP  \t fp  \t tn  \t fn')
    print('{:.0f} \t {:.0f} \t {:.0f} \t {:.0f} \t'.format(tps, fps, tns, fns))
    
    print()
    print('Relation error count: {:.0f}'.format(total_r_error))
    
        
    if rel_detail==True:
#         check_true, check_pred = check_every_rel(y_rel_true_all, y_rel_pred_all, len(schema['relation']))
#         show_every_rel_score(check_true, check_pred, schema)
        

        all_er_score = precision_recall_fscore_support(y_rel_true_all, y_rel_pred_all, average=None, 
                                           labels=range(len(schema['relation'])))[:-1]

        
        for num_rel in range(len(schema['relation'])):
            
            print()
            print('======================================================')
            print('Relation type %d' % (num_rel))
            print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
            print('%.3f \t\t %.3f \t\t %.3f \t' % (all_er_score[0][num_rel], all_er_score[1][num_rel], all_er_score[2][num_rel]))
            print()
            
    if analyze:
        draw_zone_distance(anay_rel_true, anay_rel_pred, anay_rel_pospred, schema)
        
                
    return e_score, er_score
                
                
                
def batch_decode(ent_output, rel_output, batch_index, word_lists, true_ent, true_rel, schema, silent, analyze):
    
    true_ent_lists, true_rel_lists = [], []
    pred_ent_lists, pred_rel_lists = [], []
    rel_error_count = 0

    anay_true, anay_pred, anay_pospred = {}, {}, {}    # add postive relation
    for i in range(len(schema['relation'])):
        anay_true[i]=[]
        anay_pred[i]=[]
        anay_pospred[i]=[]
    
    for e,r,i,te,tr in zip(ent_output, rel_output, batch_index, true_ent, true_rel):
        
        # 算句子長度
        len_of_list = len(word_lists[i])
        word_list = word_lists[i]
        
        true_ent = [schema.ent2ix.inv(i) for i in te[:len_of_list]]
        predict_ent = [schema.ent2ix.inv(i) for i in ent_argmax(e[:len_of_list])]

        
        true_ent_list, _ = decode_ent(te[:len_of_list], schema)
        pred_ent_list, _ = decode_ent(e[:len_of_list], schema)
        
     
        true_r_list, appear_error = decode_rel(true_ent, tr, schema)  
        pred_r_list, appear_error = decode_rel(predict_ent, r, schema)      
        
        # 出現error，跳過這句
        if appear_error:
            rel_error_count+=1
#             continue
        
        
        true_r_list = [list(set(i)) if type(i) is list else i for i in true_r_list]
        pred_r_list = [list(set(i)) if type(i) is list else i for i in pred_r_list]
        
        true_r_list = true_r_list[:len_of_list]
        pred_r_list = pred_r_list[:len_of_list]
        
        
        true_rel_list = decode_rel_to_eval(true_r_list, schema, true_ent_list)
        pred_rel_list = decode_rel_to_eval(pred_r_list, schema, pred_ent_list)
        

        
        true_ent_lists.append(true_ent_list)
        pred_ent_lists.append(pred_ent_list)
        true_rel_lists.append(true_rel_list)
        pred_rel_lists.append(pred_rel_list)


        
        if not silent:
            print(word_list)
            print(true_ent)
            print(true_r_list)
            print()
            print('Predict output')
            print(predict_ent)
            print(pred_r_list)
            print()
            print('True')
            print(true_ent_list)
            print(true_rel_list)
            print('predict')
            print(pred_ent_list)
            print(pred_rel_list)
            print("=====================================")
            
        
        if analyze:
            postive_predict_rel = list(set(true_rel_list).intersection(pred_rel_list))
            
            analyze_dict_true = calculate_distance(true_rel_list)   
            analyze_dict_pred = calculate_distance(pred_rel_list) 
            analyze_dict_pos_pred = calculate_distance(postive_predict_rel) 
            
            for r in analyze_dict_true:
                anay_true[r].extend(analyze_dict_true[r])
            for r in analyze_dict_pred:
                anay_pred[r].extend(analyze_dict_pred[r])
            for r in analyze_dict_pos_pred:
                anay_pospred[r].extend(analyze_dict_pos_pred[r])
                
            
        


        
    ent_score, y_true_ent, y_pred_ent = get_scores(true_ent_lists, pred_ent_lists, 
                                                    range(len(schema['entity'])), output_y=True)
    
    rel_score, y_true_rel, y_pred_rel = get_scores(true_rel_lists, pred_rel_lists, 
                                                    range(len(schema['relation'])), output_y=True)
        
    
    tp, fp, tn, fn = relation_error_analysis(true_rel_lists, pred_rel_lists)
    
    if not silent:
        print('Batch entity score')
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print(ent_score)
        print()
        print('Batch relation score')
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print(rel_score)
        print()
        print('p_r_fscore')
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print(p_r_fscore(tp, fp, tn, fn), tp, fp, tn, fn)
        print('===========================================') 
    
    
    return anay_true, anay_pred, anay_pospred, rel_error_count, y_true_ent, y_pred_ent, y_true_rel, y_pred_rel, tp, fp, tn, fn        



def ent_argmax(output):
    return output.argmax(-1)
                
def rel_argmax(output):
    output = output.argmax(-1)
    return output              
                
                
def decode_ent(ent_output, schema):
    '''
    Aggregate entities from predicted tags
    Input:
    pred_ent=a list of entity tags in a sentence
    schema=the dictionary defining entities and relations
    Output: 
    ent_list=[(ent_start, ent_end, ent_type=eid_in_schema)]
    err_count=the number of bad tags
    '''
    ent_list = []
    ent_start = 0
    ent_end = 0
    state = {
        'ENT_SPAN': 0,
        'NO_ENT': 1
    }
    err_count = 0
    
    if len(ent_output.size()) == 2:
        ent_output = ent_argmax(ent_output)
    
    ent_type = ''
    sid = state['NO_ENT']
    
    for idx, e_idx in enumerate(ent_output):
        bio = schema.ent2ix.inv(e_idx)[0]

        
        if sid == state['NO_ENT']:
            if bio == 'B':
                ent_start = idx
                ent_type = schema.eid_from_tag_ix(e_idx)
                sid = state['ENT_SPAN']
                
            elif bio == 'I':
                err_count += 1
                
        elif sid == state['ENT_SPAN']:
            if bio != 'I':
                ent_end = idx - 1
                ent_list.append((ent_start, ent_end, ent_type))
                
                if bio == 'B':
                    ent_start = idx
                    ent_type = schema.eid_from_tag_ix(e_idx)
                    
                else:
                    sid = state['NO_ENT']
                    
            elif ent_type != schema.eid_from_tag_ix(e_idx):
                ent_end = idx - 1
                ent_list.append((ent_start, ent_end, ent_type))
                err_count += 1
                sid = state['NO_ENT']
                
    if sid == state['ENT_SPAN']:
        ent_end = ent_output.size(0) - 1
        ent_list.append((ent_start, ent_end, ent_type))
        
    return ent_list, err_count




def decode_rel(ent_output, rel_output, schema):

    r_list, r_dict, appear_error = create_rel_info(ent_output) 
    
    IsB = False           # 是否遇到B tag的lock
    IsNext = False        # 是否為B tag後面的tag 的lock
    num_reocrd = -1       # 紀錄pair數
    now_loc = 0
    pre_rel_end_loc = 0
    now_rel_end_loc = 0
    
    rel_keyerror = False

    if len(rel_output.size())==3:
        rel_output = rel_argmax(rel_output)
    
    
    for now in range(len(rel_output)):
        for loc, rel in enumerate(rel_output[now][:now+1]):
            rel = rel.cpu().numpy()
            
#             print(rel, IsB, IsNext)

            
            # 有關係存在，且為B tag 
            if rel!=schema.rel2ix[schema.REL_NONE] and IsB==False and IsNext==False:

                IsB = True
                IsNext = True
                
                tag = schema.rel2ix.inv(int(rel))
                num_reocrd+=1
                now_loc = loc
                
                # 錯誤來自於，now_loc找不到，也就是說，rel預測出來是有關係存在
                # 但預測是'O'
                # 而在entity中卻沒有預測出來，所以r_dict中沒有紀錄
                try:
                    pre_rel_end_loc = r_dict[now_loc]['end']

                except KeyError:
                    rel_keyerror = True
                    break
                    
                
                try:
                    now_rel_end_loc = r_dict[now]['end']
                    
                except KeyError:
                    rel_keyerror = True
                    break
            
             
                second_tag = r_dict[now_loc]['_2ndtag']
                preAorB = check_rel_loc(second_tag, schema)
                nowAorB = 'B' if preAorB=='A' else 'A'
                
                pre_complete_rel = tag+"-"+str(num_reocrd)+"-"+preAorB
                now_complete_rel = tag+"-"+str(num_reocrd)+"-"+nowAorB
                
                # 將以前的token填上關係
                for token in range(now_loc, pre_rel_end_loc+1):
                    
                    # 出現以下error
                    '''AttributeError: 'str' object has no attribute 'append'''
                    # 為 r_list 前處理中沒有給予可能有關係的位置空的list
                    try:
                        r_list[token].append(pre_complete_rel)
                    except AttributeError:
                        r_list[token] = []
                        r_list[token].append(pre_complete_rel)

             
                # 當前token填上關係
                r_list[now].append(now_complete_rel)
                

            
            # 關係前位中B tag後面的tag
            elif rel!=schema.rel2ix[schema.REL_NONE] and IsB:
                # 如果還在這個entity的範圍內
                if loc<=pre_rel_end_loc:
                    pass
                
                # 超出現在這個entity的範圍，改lock
                else:
                    IsB = False

            
            # B tag後面的tag的關係，依照前面的關係複製
            elif rel!=schema.rel2ix[schema.REL_NONE] and IsNext:
                
                # IndexError: list assignment index out of range
                try:
                    r_list[now] = r_list[now-1]
                except IndexError:
                    rel_keyerror = True   # 暫時沿用keyerror
                    break
                
                
                
            else:
                if now<=now_rel_end_loc:
                    IsB = False
                else:
                    IsB = False
                    IsNext = False

        
        if rel_keyerror:
            appear_error = True
            break
                
                
    return r_list, appear_error
                
    
    
def create_rel_info(ent_output):
    r_list = []     # 存放完整關係
    r_dict = {}     # 記錄關係資訊
    appear_error = False
    

    e_loc = 0       # 當前遇到的entity的位置
    for loc, e in enumerate(ent_output):
        if e[0]=='B':
            e_loc = loc
            r_dict[loc] = {
                '_2ndtag':e[2:],
                'end':loc,
            }
            r_list.append([])
                   
        elif e[0]=='I':
            # 錯誤來自於，entity預測錯誤，沒有預測到B tag，直接跳到I tag
            # 所以沒有紀錄e_loc
            try: 
                r_dict[e_loc]['end'] = loc
            except KeyError:
                appear_error = True
                break
            
            r_list.append([])
            
        else:
            r_list.append("")
              
    return r_list, r_dict, appear_error
    
    

# 是三元關係中的前者還是後者                  
def check_rel_loc(second_tag, schema):
    convert_tag = ''

    for ent_content in schema['entity']:
        if schema['entity'][ent_content]['tag']==second_tag:
            convert_tag = ent_content
    
    rel_types = schema['relation'].values()

    for rel_content in schema['relation'].values():
        for AorB in rel_content['arguments']:
            if rel_content['arguments'][AorB]==convert_tag:
                return AorB    



                
def decode_rel_to_eval(r_list, schema, ent_list):
    
    max_pair = 0
    for r in r_list:
        if type(r) is list:
            for single_r in r:
                if int(single_r[-3])>max_pair:
                    max_pair = int(single_r[-3])
    

    pair_idx = {}
    for pair in range(max_pair+1):
        for i, r in enumerate(r_list):
            if type(r) is list:
                for single_r in r:
                    if int(single_r[-3])==pair:
                        if pair not in pair_idx:
                            pair_idx[pair] = [i]
                        else:
                            pair_idx[pair].append(i)
    
 

    pair_list = []
    for i, e_pair_1 in enumerate(ent_list):
        for j, e_pair_2 in enumerate(ent_list[i+1:]):
            if e_pair_1[-1]!=e_pair_2[-1]:
                pair_list.append([e_pair_1, e_pair_2])
                
                
    
    
    eval_rel_list = []
    for pair in pair_idx:
        rel_loc = pair_idx[pair]
        for e_pairs in pair_list:
            check_pair = True
            first_start = e_pairs[0][0]
            first_end = e_pairs[0][1]
            sec_start = e_pairs[1][0]
            sec_end = e_pairs[1][1]
            
            first_l = list(range(first_start, first_end+1))
            sec_l = list(range(sec_start, sec_end+1))
            combine_l = first_l+sec_l

            
            check_in_entity = []
            for x in combine_l:
                if x in rel_loc:
                    check_in_entity.append(True)
                else:
                    check_in_entity.append(False)
                    check_pair = False
                    break             

            if all(check_in_entity)==True:
                for r in r_list[first_start]:
                    r_info = r.split('-')
                    
                    if int(r_info[1])==pair:
                        r_tag = get_rid_from_tag(r_info[0], schema)
                        e_pairs_copy = e_pairs
                        e_pairs_copy.append(r_tag)
                        eval_rel_list.append(tuple(e_pairs_copy))
    
    return eval_rel_list
    
        
def get_rid_from_tag(tag, schema):   
    for content in schema['relation'].values():
        if content['tag'] == tag:
            return content['rid']





        
        
def get_scores(true_lists, pred_lists, labels, output_y=False):
    y_true, y_pred = [], []
    for t_list, p_list in zip(true_lists, pred_lists):
        yt, yp = align_yt_yp(t_list, p_list, labels)
        y_true.extend(yt)
        y_pred.extend(yp)
        
    scores = precision_recall_fscore_support(y_true, y_pred, average='micro', labels=labels)
    return scores, y_true, y_pred if output_y else scores

def align_yt_yp(truths, predictions, labels):
    '''
    Input:
        truths/predictions: list of true and predicted tuples, 
        with the leading entries as the structure and the last entry as the class,
        e.g., [(e1, e2, rel), ...]
        labels: sequence of valid class
    Output:
        yt: list of true class given a structure
        yp: list of predicted class given a structure
    '''
    yt, yp = [], []
    _ID_NONE = len(labels)
    true_dict = { t[:-1]: t[-1] for t in truths }
    for p in predictions:
        yt.append(true_dict.pop(p[:-1], _ID_NONE))
        yp.append(p[-1])
    for target in true_dict.values():
        yt.append(target)
        yp.append(_ID_NONE)

    return yt, yp



def is_neg_triple(t):
    return np.imag(t[-1]) > 0

def negate_triple(t):
    # Mark negative triples with imaginary relation id
    return (t[0], t[1], np.real(t[-1]).item() + 1j)

def posit_triple(t):
    return (t[0], t[1], np.real(t[-1]).item())

def has_edge(base_ptrs, rel, e):
    '''
    Assume a relation exist between an entity pair, 
    if all the tokens in the base entity point to those in entity e.
    '''
    tok_has_ptr_to_e = [tok_ptrs[rel].ge(e[0]).dot(tok_ptrs[rel].le(e[1])).item() > 0 
                        for tok_ptrs in base_ptrs]
    return len(tok_has_ptr_to_e) > 0 and all(tok_has_ptr_to_e)


def relation_error_analysis(true_rel_lists, rel_lists):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, r_list in enumerate(rel_lists):
        true_pos = len([t for t in r_list if t in true_rel_lists[i]])
        all_true = len([t for t in true_rel_lists[i] if not is_neg_triple(t)])
        all_pos = len(r_list)
        tp += true_pos
        fn += all_true - true_pos
        fp += all_pos - true_pos
        tn += len([t for t in true_rel_lists[i] if is_neg_triple(t) and posit_triple(t) not in r_list])
    return tp, fp, tn, fn

def p_r_fscore(tp, fp, tn, fn, beta=1, eps=1e-8):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f_beta = (1 + beta**2) * ((p * r) / (((beta**2) * p) + r + eps))
    return p, r, f_beta


def check_every_rel(rel_true, rel_pred, none_rel):
    
    check_true = []
    check_pred = []
    for i in range(none_rel):
        check_true.append([])
        check_pred.append([])
   
    for r_t,r_p in zip(rel_true, rel_pred):
        if r_t==r_p:
            check_true[r_t].append(r_t)
            check_pred[r_p].append(r_p)
        
        elif r_t==none_rel:
            check_true[r_p].append(r_t)
            check_pred[r_p].append(r_p)
        
        elif r_p==none_rel:
            check_true[r_t].append(r_t)
            check_pred[r_t].append(r_p)
    
# #         看關聯分類後的結果
# #         上面出現_ID_NONE是誤報
# #         下面出現_ID_NONE是漏報
#     print(check_true)
#     print(check_pred)

    
    return check_true, check_pred
        

def show_every_rel_score(check_true, check_pred, schema):
    
    for num_rel in range(len(check_true)):
        each_scores = precision_recall_fscore_support(check_true[num_rel], \
                    check_pred[num_rel], average='micro', labels = range(len(schema['relation'])))
        
        
        print()
        print('======================================================')
        print('Relation type %d' % (num_rel))
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print('%.3f \t\t %.3f \t\t %.3f \t' % (each_scores[0], each_scores[1], each_scores[2]))
        print()


        
# ===================================================================        
        
        
def analyze_loader(data_loader, schema, silent=False):
    
    all_rel = {}
    for i in range(len(schema['relation'])):
        all_rel[i]=[]
    
    for batch_x, batch_ent, batch_rel, batch_index in data_loader:
        batch_rel, rel_error_count = analyze_batch(batch_index, data_loader.raw_input,batch_ent.cpu(), 
                                                  batch_rel.cpu(), schema, silent=silent)
        
        for r in batch_rel:
            all_rel[r].extend(batch_rel[r])
    
#     print(all_rel)
    
    return all_rel
        
        
        
def analyze_batch(batch_index, word_lists, true_ent, true_rel, schema, silent):
    
    rel_error_count=0
    batch_rel = {}
    for i in range(len(schema['relation'])):
        batch_rel[i]=[]
  
    
    for i,te,tr in zip(batch_index, true_ent, true_rel):
         # 算句子長度
        len_of_list = len(word_lists[i])
        word_list = word_lists[i]
        
        true_ent = [schema.ent2ix.inv(i) for i in te[:len_of_list]]     
        true_ent_list, _ = decode_ent(te[:len_of_list], schema)   
        true_r_list, appear_error = decode_rel(true_ent, tr, schema)  
        
        # 出現error，跳過這句
        if appear_error:
            rel_error_count+=1
            continue
            
        true_r_list = [list(set(i)) if type(i) is list else i for i in true_r_list]
        true_r_list = true_r_list[:len_of_list]
        true_rel_list = decode_rel_to_eval(true_r_list, schema, true_ent_list)
     
        
        if not silent:
            print(word_list)
            print(true_ent)
            print(true_r_list)
            print()

            print()
            print('True')
            print(true_ent_list)
            print(true_rel_list)
            
            print("=====================================")
            
        analyze_dict = calculate_distance(true_rel_list)    
        for r in analyze_dict:
            batch_rel[r].extend(analyze_dict[r])
            
#     print(batch_rel)
    return batch_rel, rel_error_count
        
        
        
        
            
            
def calculate_distance(true_rel_list):
    analyze_dict = {}
    
    for r_triplet in true_rel_list:
        distant = r_triplet[1][0] - r_triplet[0][0]
        r_type = r_triplet[2]
        
        if r_type in analyze_dict:
            analyze_dict[r_type].append(distant)
        else:
            analyze_dict[r_type] = [distant]
        
        
#     print(analyze_dict)
    
    return analyze_dict      
        
        


def draw_zone_distance(anay_rel_true, anay_rel_pred, anay_rel_pospred, schema):
    
    zone_block_list = ['1~5', '6~10', '11~15', '16~20', '21~30', '31~40', '41~50', '50up']
    
    plt.subplots(3,2,figsize=(15,15))
    for r_type in anay_rel_true:
        
        zone_block_t = record_zone(anay_rel_true, r_type)
        zone_block_p = record_zone(anay_rel_pred, r_type)
        zone_block_pp = record_zone(anay_rel_pospred, r_type)  # postive predict

        
        print(zone_block_t)
        print(zone_block_p)
        print(zone_block_pp)
        print()
        
        plt.subplot(320+r_type+1)

        for i, block_range in enumerate(zone_block_list):
            t_bar = plt.bar(i-0.2, zone_block_t[block_range], facecolor='#9999ff', edgecolor='white', width=0.5) 
            p_bar = plt.bar(i, zone_block_p[block_range], facecolor='#FF8888', edgecolor='white', width=0.5) 
            pp_bar = plt.bar(i+0.2, zone_block_pp[block_range], facecolor='#48D1CC', edgecolor='white', width=0.5) 
            
            
#             plt.text(i, zone_block_t[block_range], 
#                      '{:.2f} %'.format(zone_block_t[block_range]/len(anay_rel_true[r_type])*100), ha='center', va= 'bottom')
          
#             if len(anay_rel_pred[r_type])==0:
#                 len_of_anay_rel_pred = 1
#             else:
#                 len_of_anay_rel_pred = len(anay_rel_pred[r_type]) 
#             plt.text(i+0.2, zone_block_p[block_range], 
#                      '{:.2f} %'.format(zone_block_p[block_range]/len_of_anay_rel_pred*100), ha='center', va= 'bottom')
            
            
            
        plt.xticks(range(len(zone_block_list)), zone_block_list) 
        plt.ylim(top=plt.ylim()[1]+5)
        plt.xlabel('The entity pair\'s distance in {}'.format(schema.rid2tag[r_type]))
        plt.ylabel('The number of entity pair in the zone')
        plt.legend((t_bar[0], p_bar[0], pp_bar[0]), ('True relation', 'Predict relation', 'True Predict relation'))
        
        
def record_zone(anay_rel, r_type):
    zone_block = {'1~5':0, '6~10':0, '11~15':0, '16~20':0, 
                  '21~30':0, '31~40':0, '41~50':0, '50up':0}
    
    for r_dist in anay_rel[r_type]:
        if r_dist<=5:
            zone_block['1~5']+=1
        elif r_dist<=10:
            zone_block['6~10']+=1
        elif r_dist<=15:
            zone_block['11~15']+=1
        elif r_dist<=20:
            zone_block['16~20']+=1
        elif r_dist<=30:
            zone_block['21~30']+=1
        elif r_dist<=40:
            zone_block['31~40']+=1
        elif r_dist<=50:
            zone_block['41~50']+=1
        else:
            zone_block['50up']+=1
            
    return zone_block