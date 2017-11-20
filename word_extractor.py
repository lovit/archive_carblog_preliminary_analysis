def make_document_frequency_distribution(corpus_fnames, tokenizer):
    dfs = {}
    for n_corpus, fname in enumerate(corpus_fnames):        
        corpus_index = int(fname.split('/')[-1].split('.')[0])
        corpus = DoublespaceLineCorpus(fname, iter_sent=False)
        df = _tokenize(corpus, tokenizer)
        dfs[corpus_index] = df
    _make_document_frequency_distribution(dfs, tokenizer):

def _tokenize(corpus, tokenizer):
    df = {}
    for n_doc, doc in enumerate(corpus):
        words = {word for token in doc.split() for word in tokenize(token) if len(word) > 1}
        for word in words:
            df[word] = df.get(word, 0) + 1
        if n_doc % 1000 == 999:
            print('\r  - tokenizing ... {} docs'.format(n_doc+1), flush=True, end='')
    return df

def make_subword_slot(dfs, tokenizer):
    subword2index = {subword:index for index, subword in enumerate(sorted(tokenizer.get_words()))}
    n = len(subword2index)
    m = len(dfs)
    subword_slot = np.zeros((n, m), dtype=np.float16)
    
    for j in range(m):
        for subword, df_ij in dfs[j].items():
            if not (subword in subword2index):
                continue
            i = subword2index[subword]
            df_ij = df_ij / num_doc
            subword_slot[i,j] = df_ij
    
    return subword_slot, subword2index