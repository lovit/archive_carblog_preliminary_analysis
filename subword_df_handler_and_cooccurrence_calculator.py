class SubwordDocumentFrequencyHandler:
    def __init__(self, corpus_directory, parameter_fname):
        with open(parameter_fname, 'rb') as f:
            import pickle
            params = pickle.load(f)
        self.subword_slot = params['df']
        self.index2subword = params['index2subword']
        self.subword2index = {word:index for index,word in enumerate(self.index2subword)}
        self.corpus_length = self._corpus_length(corpus_directory)
        self.num_words, self.num_categories = self.subword_slot.shape
        
    def _corpus_length(self, corpus_directory):
        import glob
        files = glob.glob(corpus_directory + '*.txt')
        files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        corpus_length = []
        for file in files:
            with open(corpus_directory+'/'+file, encoding='utf-8') as f:
                corpus_length.append(len(f.readlines()))
        return corpus_length
    
    def decode(self,idx):
        if 0 <= idx < self.num_words:
            return self.index2subword[idx]
        return None
    
    def get_total_df_ratio_from_word(self, word):
        idx = self.subword2index.get(word, -1)
        return self.get_total_df_ratio_from_word_index(idx)
        
    def get_total_df_ratio_from_word_index(self, idx):
        if idx == -1 : return None
        total_freq = [w*l for w,l in zip(self.subword_slot[idx], self.corpus_length)]
        return 100*sum(total_freq)/sum(self.corpus_length)
    
    def total_df_ratio_for_all_words(self):
        total_df_ratio = []
        for i in range(self.num_words):
            print('\r  computing total df ratio {} / {}'.format(i+1, self.num_words), flush=True, end='')
            total_df_ratio.append(self.get_total_df_ratio_from_word_index(i))
        print('\r total df ratio computing was done.        ')
        return total_df_ratio
        
def get_positive_words(positive_corpus,
                       positive_total_df_ratio,
                       reference_corpus,
                       reference_total_df_ratio,
                       min_percentage_of_positive_words,
                       min_percentage_of_reference_words):
    def is_int(word):
        try:
            word = int(word)
            return True
        except:
            return False
    positive_words = set([w for w, r in zip(positive_corpus.index2subword, positive_total_df_ratio) if r > min_percentage_of_positive_words and not is_int(w)])
    reference_words = set([w for w, r in zip(reference_corpus.index2subword, reference_total_df_ratio) if r > min_percentage_of_reference_words and not is_int(w)])
    filtered_positive_words = positive_words - reference_words
    
    return filtered_positive_words

def index_to_categories(directory):
    with open(directory, 'r', encoding='utf-8') as f:
        categories = f.readlines()
        categories = [c.strip() for c in categories]
    return categories

def get_category_sensitive_words_list(positive_corpus, index_to_categories, max_average_ratio):
    category_sensitive_words = {i:[] for i in range(len(index_to_categories))}
    for index, df_dist in enumerate(positive_corpus.subword_slot):
        if df_dist.max()/df_dist.mean() > max_average_ratio:
            category_index = df_dist.argmax()
            word = positive_corpus.decode(index)
            category_sensitive_words[category_index].append(word)
    category_sensitive_words = [set(words) for c, words in sorted(category_sensitive_words.items())]
    return category_sensitive_words

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