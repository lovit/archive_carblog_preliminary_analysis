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