class SubwordDocumentFrequencyHandler:
    def __init__(self, parameter_fname):
        with open(parameter_fname, 'rb') as f:
            import pickle
            params = pickle.load(f)
        self.subword_slot = params['df']
        self.index2subword = params['index2subword']
        self.subword2index = {word:index for index,word in enumerate(self.index2subword)}
        self.num_words, self.num_categories = self.subword_slot.shape
        
    def decode(self, idx):
        if 0 <= idx < self.num_words:
            return self.index2subword[idx]
        return None
    
    def encode(self, subword):
        return self.subword2index.get(subword, -1)
    
    def get_df_distribution_statistics_from_word(self, word):
        return self.get_df_distribution_statistics_from_word_index(self.encode(word))
    
    def get_df_distribution_statistics_from_word_index(self, idx):
        if idx == -1: return None
        df_dist = self.subword_slot[idx,:]
        return df_dist.std() / df_dist.mean(), df_dist.mean(), df_dist.max() / df_dist.mean(), df_dist.argmax()
    
    def get_df_distribution_statistics_of_all_words(self):
        nstd_mean = {}
        for idx in range(self.num_words):
            if idx % 1000 == 999:
                print('\r  computing (std/mean, mean, max/mean, argmax) {} / {}'.format(idx+1, self.num_words), flush=True, end='')
            nstd_mean[self.decode(idx)] = self.get_df_distribution_statistics_from_word_index(idx)
        print('\r computing (std/mean, mean, max/mean, argmax) was done.        ')
        return nstd_mean
        
def get_positive_words(pos_nstd_mean,
                       ref_nstd_mean,
                       pos_max_df_nstd=1.0,
                       pos_min_df_mean=0.005,
                       ref_max_df_nstd=1.5,
                       ref_min_df_mean=0.002):
    
    pos_positive = dict(filter(lambda x:(x[1][0]<pos_max_df_nstd and x[1][1]>pos_min_df_mean), pos_nstd_mean.items()),key=lambda x:x[1])
    ref_positive = dict(filter(lambda x:(x[1][0]<ref_max_df_nstd and x[1][1]>ref_min_df_mean), ref_nstd_mean.items()),key=lambda x:x[1])
    pos_filtered = {word:score for word, score in pos_positive.items() if not (word in ref_positive)}
    return pos_positive, ref_positive, pos_filtered

def pprint_word_list(word_list, cell_len=8, n_cols=5):
    n = len(word_list)
    form = '%{}s'.format(cell_len)
    for i in range(round(n/n_cols)):
        print('\t'.join([form % w for w in word_list[n_cols*i: n_cols*(i+1)]]))

def get_category_sensitive_words(pos_handler,
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


def is_co_occurred(wc, wf, doc):
    return 1 if ' '+wc in ' '+doc and ' '+wf in ' '+doc else 0


def calculate_cooccurrence_document_frequency_ratio(category_sensitive_words_list, positive_words, subword2index, corpus_directory):
    """
    Arguments:
    ----------
        category_sensitive_words_list: list of set of str
            list[category_index]: [{word, word, ....}, {word, word, ....}, ...]
            len(category_sensitive_words_list) is equal number of categories
        positive_words: set of str
        
    Returns:
    ----------
        list of sparse matrices
    """
    
    list_of_sparse_matrices = []    
    
    from collections import defaultdict
    
    for c, category_sensitive_words in enumerate(category_sensitive_words_list):
        if len(category_sensitive_words) == 0:
            print('category = {}, There are no category sensitive words here'.format(c))
            continue
        wc_occurrence = defaultdict(int) 
        cooccurrence = defaultdict(lambda: defaultdict(int))
        corpus_fname = '{}/{}.txt'.format(corpus_directory, c)

        with open(corpus_fname, 'r', encoding='utf-8') as f:
            postings = f.readlines()
        process_time = time.time()
        for i_wc, wc in enumerate(category_sensitive_words):
            filtered_postings = [post for post in postings if wc in post]
            for wf in positive_words:     
                cooccur = sum([is_co_occurred(wc, wf, post) for post in filtered_postings]) / len(filtered_postings)
                cooccurrence[wc][wf] = cooccur
            if i_wc == 0:
                continue
            print('\r scanned {}/{} category sensitive words'.format(i_wc+1, len(category_sensitive_words)), flush=True, end='')
        
        process_time = time.time() - process_time
        print('\rcategory = {}, processing time = {}'.format(c, '%.2f sec' % process_time))
        
        
        from scipy.sparse import csr_matrix
        row_ind = []
        col_ind = []
        data = []
        for sensitive_word, positive_word_counter in cooccurrence.items():
            i = subword2index.get(sensitive_word, -1)
            if i == -1:
                continue
            for positive_word, cooccur in positive_word_counter.items():
                j = subword2index.get(positive_word, -1)
                if j == -1:
                    continue
                row_ind.append(i)
                col_ind.append(j)
                data.append(cooccur)
            
        csr_matrix((data, (row_ind, col_ind)))
    
    list_of_sparse_matrices.append(csr_matrix)
        
    return list_of_sparse_matrices