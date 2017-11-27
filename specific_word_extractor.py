from collections import defaultdict
import datetime
import time
import os
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

# Step 2. Extracting sensitive words for each category
def extract_category_sensitive_words(pos_handler,
                                 pos_statistics,
                                 pos_min_df_nstd=2.5,
                                 pos_df_max_mean=0.01,
                                 pos_min_of_df_mean_ratio=3
                                ):
    word_by_category = [[] for _ in range(pos_handler.num_categories)]
    for subword, (nstd, mean, topmean, max_sensitive_category) in pos_statistics.items():
        if nstd < pos_min_df_nstd:
            continue
        if mean > pos_df_max_mean:
            continue
        idx = pos_handler.encode(subword)
        df_dist = pos_handler.subword_slot[idx,:]
        sensitive_categories = [i for i, r in enumerate(df_dist/mean) if r >= pos_min_of_df_mean_ratio]
        for c in sensitive_categories:
            word_by_category[c].append(subword)
    return word_by_category

# Step 3. Extracting specpfic words for each category
def does_cooccurred(w_sens, w_pos, doc):
    return (w_sens in doc) and (w_pos in doc)

def _calculate_cooccurrence_df(D_sensitive, D_positive, corpus_fname, debug=False):
    D_positive = {' '+word.strip() for word in D_positive}
    D_sensitive = {' '+word.strip() for word in D_sensitive}
    cooccurrence_dd = defaultdict(lambda: defaultdict(int))
    df_sensitive = defaultdict(int)
    
    with open(corpus_fname, encoding='utf-8') as f:
        for i_doc, doc in enumerate(f):
            doc = ' ' + doc
            for i_ws, w_sens in enumerate(D_sensitive):
                if debug and i_ws >= 199: break
                if w_sens in doc:
                    df_sensitive[w_sens] += 1
                else:
                    continue
                for w_pos in D_positive:
                    if not does_cooccurred(w_sens, w_pos, doc):
                        continue
                    cooccurrence_dd[w_sens][w_pos] += 1
            if debug and i_doc >= 500: break
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

def calculate_cooccurrence(D_sensitive_by_category, D_positive, subword2index, corpus_directory, model_directory, debug=False):
    for c, D_sensitive in enumerate(D_sensitive_by_category):
        if debug and c >= 3: 
            break
    
        process_time = time.time()
        corpus_fname = '{}/{}.txt'.format(corpus_directory, c)
        cooccurrence_dd_norm, df_sensitive = _calculate_cooccurrence_df(D_sensitive, D_positive, corpus_fname, debug)
        process_time = time.time() - process_time
        process_time = str(datetime.timedelta(seconds=process_time))
        print('\rcalculating cooccurrence in {} / {} corpus ({})'.format(c, len(D_sensitive_by_category), process_time), flush=True)
        if not cooccurrence_dd_norm:
            continue
        cooc_fname = '{}/cooccurrance_c{}.mtx'.format(model_directory, c)
        df_fname = '{}/df_c{}.csv'.format(model_directory, c)
        _save_dictdict_as_sparsematrix(cooccurrence_dd_norm, subword2index, cooc_fname)
        _save_df_as_list(df_sensitive, subword2index, df_fname)