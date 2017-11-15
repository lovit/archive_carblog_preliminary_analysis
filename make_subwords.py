import argparse
from glob import glob
import pickle
import numpy as np

def parse_subwords(corpus_fnames, model_directory, soynlp_path):
    import sys
    sys.path.append(soynlp_path)
    from soynlp.utils import DoublespaceLineCorpus
    from soynlp.utils import get_process_memory
    
    for n_corpus, fname in enumerate(corpus_fnames):
        corpus_index = fname.split('/')[-1].split('_')[0]
        model_fname = '{}/{}_subword_statistics.pkl'.format(model_directory, corpus_index)
        corpus = DoublespaceLineCorpus(fname, iter_sent=False)

        L = {}
        DF = {}
        for n_doc, doc in enumerate(corpus):
            for word in doc.split():
                for e in range(1, max(8, len(word))+1):
                    l = word[:e]
                    L[l] = L.get(l, 0) + 1
            subwords = {word[:e] for word in set(doc.split()) for e in range(2, max(8, len(word))+1) if len(word) > 1}
            for subword in subwords:
                DF[subword] = DF.get(subword, 0) + 1
            if n_doc % 1000 == 999:
                print('\rscanning ... {} / {}, {} docs'.format(n_corpus, len(corpus_fnames), n_doc+1), flush=True, end='')

        cohesions = {}
        for l, count in L.items():
            n = len(l)
            if n < 2 or count < 10:
                continue
            cohesion = pow(count/L[l[0]], 1/(n-1))
            cohesions[l] = cohesion

        params = {
            'l_frequency':L,
            'l_document_frequency':DF,
            'l_cohesion': cohesions,
            'num_doc': (n_doc+1)
        }
        print('\rscanning was done {} / {}, used memory = {} Gb'.format(n_corpus+1, len(corpus_fnames), ' %.3f'%get_process_memory()))
        with open(model_fname, 'wb') as f:
            pickle.dump(params, f)

        del cohesions
        del DF
        del L
    print('subword parsing was done')

def make_universial_vocabulary(corpus_fnames, model_directory, min_frequency)
    universial_subwords = set()
    for n_corpus, fname in enumerate(corpus_fnames):
        corpus_index = fname.split('/')[-1].split('_')[0]
        model_fname = '{}/{}_subword_statistics.pkl'.format(model_directory, corpus_index)
        with open(model_fname, 'rb') as f:        
            params = pickle.load(f)
            for subword, frequency in params['l_frequency'].items():
                if frequency < min_frequency or len(subword) < 2:
                    continue
                universial_subwords.add(subword)
        print('cumulated {} corpus, {} subwords'.format(n_corpus+1, len(universial_subwords)))
    print('done')

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
            for subword, df in params['l_document_frequency'].items():
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
    print('done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_directory', type=str, default='./', help='corpus directory')
    parser.add_argument('--model_directory', type=str, default='./', help='model directory')
    parser.add_argument('--soynlp_path', type=str, default='./', help='soynlp package path')
    parser.add_argument('--min_frequency', type=str, default='./', help='minimum frequency for universial vocabulary construction')
    
    args = parser.parse_args()
    
    corpus_fnames = sorted(glob('{}/*_text'.format(args.corpus_directory)))
    model_directory = args.model_directory
    soynlp_path = args.soynlp_path
    min_frequency = args.min_frequency
    
    print('begin parsing subwords')
    parse_subwords(corpus_fnames, model_directory, soynlp_path)
    
    print('making universial vocabulary')
    universial_subwords = make_universial_vocabulary(corpus_fnames, model_directory, min_frequency)
    
    print('making subword slot')
    make_subword_slot(universial_subwords, corpus_fnames, model_directory)
    

if __name__ == "__main__":
    main()