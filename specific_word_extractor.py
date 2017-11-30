from collections import Counter
from collections import defaultdict
import datetime
import time
import os
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

# Step 3. Extracting specpfic words for each category
def does_cooccurred(w_sens, w_pos, doc):
    return (w_sens in doc) and (w_pos in doc)

def _calculate_cooccurrence_df(D_sensitive, D_positive, corpus_fname, debug=False):
    D_positive = {word.strip() for word in D_positive}
    D_sensitive = {word.strip() for word in D_sensitive}
    cooccurrence_dd = defaultdict(lambda: defaultdict(int))
    df_sensitive = defaultdict(int)
    
    max_length = max(max((len(w) for w in D_positive)), max((len(w) for w in D_sensitive))) - 1
    
    with open(corpus_fname, encoding='utf-8') as f:
        for i_doc, doc in enumerate(f):
            #doc = ' ' + doc
            subwords = {word[:e] for word in doc.split() for e in range(2, min(max_length, len(word))+1)}
            for i_ws, w_sens in enumerate(D_sensitive):
                if debug and i_ws >= 199: break
                if w_sens in subwords:
                    df_sensitive[w_sens] += 1
                else:
                    continue
                for w_pos in D_positive:
                    if w_pos in subwords:
                        cooccurrence_dd[w_sens][w_pos] += 1
            if debug and i_doc >= 500: break
            del subwords
            if i_doc % 10 == 9:
                print('\r  - calculating cooccurrence ... {} docs'.format(i_doc+1), flush=True, end='')
    
    cooccurrence_dd_normed = {}
    for w_sens, cooccurrance_dict in cooccurrence_dd.items():
        df_w_sens = df_sensitive[w_sens]
        cooccurrence_dd_normed[w_sens.strip()] = {w_pos.strip():cooc/df_w_sens for w_pos, cooc in cooccurrance_dict.items()}
    return cooccurrence_dd_normed, df_sensitive

def _save_dictdict_as_sparsematrix(dd, subword2index, fname):
    n_vocab = len(subword2index)
    row_ind = []
    col_ind = []
    data = []
    for w_sens, positive_word_counter in dd.items():
        i = subword2index.get(w_sens, -1)
        if i == -1:
            continue
        for positive_word, cooccur in positive_word_counter.items():
            j = subword2index.get(positive_word, -1)
            if j == -1:
                continue
            row_ind.append(i)
            col_ind.append(j)
            data.append(cooccur)
    x = csr_matrix((data, (row_ind, col_ind)), shape=(n_vocab, n_vocab))
    mmwrite(fname, x)

def _save_df_as_list(df, subword2index, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        for subword, _ in sorted(subword2index.items(), key=lambda x:x[1]):
            f.write('{}\n'.format(df[subword]))

def calculate_cooccurrence(category_names, D_sensitive_by_category, D_positive, subword2index, corpus_directory, model_directory, debug=False):
    for c, (category_name, D_sensitive) in enumerate(zip(category_names, D_sensitive_by_category)):
        if debug and c >= 3: 
            break
        D_positive_c = {word for word in D_positive if not (category_name in word)}
        process_time = time.time()
        corpus_fname = '{}/{}.txt'.format(corpus_directory, c)
        cooccurrence_dd_norm, df_sensitive = _calculate_cooccurrence_df(D_sensitive, D_positive_c, corpus_fname, debug)
        process_time = time.time() - process_time
        process_time = str(datetime.timedelta(seconds=process_time))
        print('\rcalculating cooccurrence in {} / {} corpus ({})'.format(c, len(D_sensitive_by_category), process_time), flush=True)
        if not cooccurrence_dd_norm:
            continue
        cooc_fname = '{}/cooccurrance_c{}.mtx'.format(model_directory, c)
        df_fname = '{}/df_c{}.csv'.format(model_directory, c)
        _save_dictdict_as_sparsematrix(cooccurrence_dd_norm, subword2index, cooc_fname)
        _save_df_as_list(df_sensitive, subword2index, df_fname)
        
# Step 3. Extracting specpfic words for each category
## (2) frequency proportion ratio score
def _subword_frequency_matrix(corpus_fname, subword_set, subword2index, c, n_categories, debug=False):
    def tokenize_subword(token, max_length=8):
        subwords = [token[:e] for e in range(2, min(len(token), max_length)+1)]
        subwords = [subword for subword in subwords if subword in subword_set]
        return subwords
    
    rows = []
    cols = []
    data = []
    with open(corpus_fname, encoding='utf-8') as f:
        for i, doc in enumerate(f):
            if i % 100 == 99:
                print('\r  - subword-tokenizing ... {} docs'.format(i+1), flush=True, end='')
            if debug and i >= 500:
                break
            subwords = Counter((subword for token in doc.split() for subword in tokenize_subword(token)))
            for subword, count in subwords.items():
                j = subword2index.get(subword, -1)
                if j == -1:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(count)
    (n, m) = (i+1, len(subword2index))
    x = csr_matrix((data, (rows, cols)), shape=(n,m))
    print('\rcreated subword frequency matrix in {} / {} corpus'.format(c, n_categories), flush=True)
    return x

def create_subword_frequency_matrix(category_names, D_sensitive_by_category, D_positive, subword2index, corpus_directory, model_directory, debug=False):
    n_categories = len(D_sensitive_by_category)
    for c, (category_name, D_sensitive) in enumerate(zip(category_names, D_sensitive_by_category)):
        if debug and c >= 3: 
            break
        
        corpus_fname = '{}/{}.txt'.format(corpus_directory, c)
        
        subword_set = {word for word in D_positive if not (category_name in word)}
        subword_set.update({word for word in D_sensitive})
        x = _subword_frequency_matrix(corpus_fname, subword_set, subword2index, c, n_categories, debug)
        x_fname = '{}/positive_subword_tf_c{}.mtx'.format(model_directory, c)
        mmwrite(x_fname, x)
        del x
        
        x = _subword_frequency_matrix(corpus_fname, {word for word in subword2index}, subword2index, c, n_categories, debug)
        x_fname = '{}/subword_tf_c{}.mtx'.format(model_directory, c)
        mmwrite(x_fname, x)