from utils import DoublespaceLineCorpus
from utils import get_process_memory
import numpy as np

def make_document_frequency_distribution(corpus_fnames, tokenizer):
    dfs = {}
    n_docs = {}
    for n_corpus, fname in enumerate(corpus_fnames):        
        corpus_index = int(fname.split('/')[-1].split('.')[0])
        corpus = DoublespaceLineCorpus(fname, iter_sent=False)
        df, n_docs_ = _tokenize(corpus, tokenizer)
        dfs[corpus_index] = df
        n_docs[corpus_index] = n_docs_
        print('\r  - tokenized {} / {} corpus, mem = {} Gb'.format(n_corpus+1, len(corpus_fnames), '%.3f'%get_process_memory()),flush=True)
    return _make_document_frequency_distribution(dfs, n_docs, tokenizer)

def _tokenize(corpus, tokenizer):
    df = {}
    for n_docs, doc in enumerate(corpus):
        words = {word for token in doc.split() for word in tokenizer.tokenize(token) if len(word) > 1}
        for word in words:
            df[word] = df.get(word, 0) + 1
        if n_docs % 1000 == 999:
            print('\r  - tokenizing ... {} docs'.format(n_docs+1), flush=True, end='')
    return df, (n_docs+1)

def _make_document_frequency_distribution(dfs, n_docs, tokenizer):
    print('make document frequency distribution table ... ', flush=True)
    subword2index = {subword:index for index, subword in enumerate(sorted(tokenizer.get_words()))}
    index2subword = [word for word, _ in sorted(subword2index.items(), key=lambda x:x[1])]
    n = len(subword2index)
    m = len(dfs)
    subword_slot = np.zeros((n, m), dtype=np.float16)
    
    for j in range(m):
        n_docs_ = n_docs[j]
        for subword, df_ij in dfs[j].items():
            if not (subword in subword2index):
                continue
            i = subword2index[subword]
            df_ij = df_ij / n_docs_
            subword_slot[i,j] = df_ij
        print('\r  - corpus {} / {} ...'.format(j+1, m), flush=True, end='')
    print('\rmaking document frequency distribution table was done ', flush=True)
    return subword_slot, index2subword