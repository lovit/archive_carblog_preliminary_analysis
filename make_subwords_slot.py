import argparse
import pickle
import sys
import numpy as np
from glob import glob

def parse_subwords(corpus_fnames, model_directory, soynlp_path, min_frequency, subword_max_length, expansion_threshold):    
    try:
        sys.path.append(soynlp_path)
        from soynlp.utils import DoublespaceLineCorpus
        from soynlp.utils import get_process_memory
    except Exception as e:
        print('importing soynlp was failed {}'.format(str(e)))
        return False
    
    for n_corpus, fname in enumerate(corpus_fnames):        
        corpus_index = fname.split('/')[-1].split('_')[0]
        corpus = DoublespaceLineCorpus(fname, iter_sent=False)
        model_fname = '{}/{}_subword_statistics.pkl'.format(model_directory, corpus_index)        
        
        L, df_subword, n_doc = _subword_counting(corpus, subword_max_length)
        cohesions, droprate_scores, word_scores = _word_scoring(L, min_frequency, expansion_threshold)
        df_word = _word_extraction(corpus, subword_max_length, word_scores)
        _save(model_fname, L, df_subword, df_word, cohesions, droprate_scores, word_scores, n_doc)
        
        print('\r  - scanning and computation were done {} / {}, used memory = {} Gb'.format(n_corpus+1, len(corpus_fnames), ' %.3f'%get_process_memory()))        
        del cohesions, df_subword, df_word, L, word_scores
        
    print('  subword parsing was done')
    return True

def _save(model_fname, L, df_subword, df_word, cohesions, droprate_scores, word_scores, n_doc):
    params = {
        'frequency':L,
        'document_frequency_subword':df_subword,
        'document_frequency_word':df_word,
        'cohesion': cohesions,
        'droprate_score': droprate_scores,
        'word_score': word_scores,
        'num_doc': n_doc
    }
    with open(model_fname, 'wb') as f:
        pickle.dump(params, f)
            
def _subword_counting(corpus, subword_max_length):
    L, df = {}, {}
    for n_doc, doc in enumerate(corpus):
        subwords = set()
        for word in doc.split():
            for e in range(1, min(subword_max_length, len(word))+1):
                subword = word[:e]
                L[subword] = L.get(subword, 0) + 1
        for subword in filter(lambda x:len(x) > 1, subwords):
            df[subword] = df.get(subword, 0) + 1            
        if n_doc % 1000 == 999:
            print('\r  - scanning ... {} docs'.format(n_doc+1), flush=True, end='')
    return L, df, (n_doc+1)

def _word_scoring(L, min_frequency, expansion_threshold):
    cohesions, droprate_scores, word_scores = {}, {}, {}
    for l, count in sorted(L.items(), key=lambda x:len(x[0])):
        n = len(l)
        if n < 2 or count < min_frequency:
            continue
        l_sub = l[:-1]
        cohesions[l] = pow(count/L[l[0]], 1/(n-1))        
        droprate = count / L[l_sub]
        droprate_scores[l_sub] = max(droprate_scores.get(l_sub, 0), droprate)
        if (l_sub in word_scores) and ((count / L[l_sub]) < expansion_threshold):
            word_scores.pop(l_sub)
        word_scores[l] = count
    droprate_scores = {word:1-score for word, score in droprate_scores.items() if len(word) > 1}
    word_scores = {word:cohesions.get(word,0) * droprate_scores.get(word,1.0) for word in word_scores}
    word_scores = {word:score for word,score in word_scores.items() if score > 0}
    return cohesions, droprate_scores, word_scores

def _word_extraction(corpus, subword_max_length, word_scores):
    def tokenize(word):
        words = [(word[:e], word_scores.get(word[:e], 0)) for e in range(2, min(subword_max_length, len(word))+1)]
        if not words:
            return ''
        best = sorted(words, key=lambda x:-x[1])[0]
        if best[1] > 0:
            return best[0]
        return ''
    
    df = {}
    for n_doc, doc in enumerate(corpus):
        words = [tokenize(word) for word in doc.split() if len(word) > 1]
        words = [word for word in words if word]
        for word in words:
            df[word] = df.get(word, 0) + 1            
        if n_doc % 1000 == 999:
            print('\r  - word extraction ... {} docs'.format(n_doc+1), flush=True, end='')
    return df

def make_universial_vocabulary(corpus_fnames, model_directory, min_frequency):
    universial_subwords = set()
    for n_corpus, fname in enumerate(corpus_fnames):
        corpus_index = fname.split('/')[-1].split('_')[0]
        model_fname = '{}/{}_subword_statistics.pkl'.format(model_directory, corpus_index)
        with open(model_fname, 'rb') as f:        
            params = pickle.load(f)
            for subword, frequency in params['document_frequency_word'].items():
                if frequency < min_frequency or len(subword) < 2:
                    continue
                universial_subwords.add(subword)
        print('  - cumulated {} corpus, {} subwords'.format(n_corpus+1, len(universial_subwords)))
    print('  done')

    with open('{}/universial_subwords.txt'.format(model_directory), 'w', encoding='utf-8') as f:
        for subword in sorted(universial_subwords):
            f.write('{}\n'.format(subword))
    return universial_subwords

def make_subword_slot(universial_subwords, corpus_fnames, model_directory):
    subword2index = {subword:index for index, subword in enumerate(sorted(universial_subwords))}
    n = len(subword2index)
    m = len(corpus_fnames)
    subword_slot = np.zeros((n, m), dtype=np.float16)
    
    for corpus_index in range(m):
        model_fname = '{}/{}_subword_statistics.pkl'.format(model_directory, corpus_index)
        with open(model_fname, 'rb') as f:
            params = pickle.load(f)
            num_doc = params['num_doc']
            for subword, df in params['document_frequency_word'].items():
                if not (subword in subword2index):
                    continue
                i = subword2index[subword]
                df = df / num_doc
                subword_slot[i,corpus_index] = df
            del params

    with open('{}/subword_df_slot.pkl'.format(model_directory), 'wb') as f:
        params = {
            'subword_slot': subword_slot,
            'subword2index': subword2index
        }
        pickle.dump(params, f)
    print('  done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_directory', type=str, default='./', help='corpus directory')
    parser.add_argument('--model_directory', type=str, default='./', help='model directory')
    parser.add_argument('--soynlp_path', type=str, default='./', help='soynlp package path')
    parser.add_argument('--min_frequency', type=int, default=100, help='minimum frequency for universial vocabulary construction')
    parser.add_argument('--subword_max_length', type=int, default=8, help='maximum length of left-side subsection (subword)')
    parser.add_argument('--expansion_threshold', type=float, default=0.95, help='minimum #(w[:-1]) / #(w)')
    
    args = parser.parse_args()
    
    corpus_fnames = sorted(glob('{}/*_text'.format(args.corpus_directory)))
    model_directory = args.model_directory
    soynlp_path = args.soynlp_path
    min_frequency = args.min_frequency
    subword_max_length = args.subword_max_length
    expansion_threshold = args.expansion_threshold

#"""
#"""
    import os
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print('created directory = {}'.format(model_directory))
    
    print('begin parsing subwords')
    parse_subwords(corpus_fnames, model_directory, soynlp_path, min_frequency, subword_max_length, expansion_threshold)
    
    print('making universial vocabulary')
    universial_subwords = make_universial_vocabulary(corpus_fnames, model_directory, min_frequency)
    
    print('making subword slot')
    make_subword_slot(universial_subwords, corpus_fnames, model_directory)
    
    print('everything was done')
"""
"""

if __name__ == "__main__":
    main()