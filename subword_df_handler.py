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