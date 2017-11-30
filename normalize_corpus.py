import argparse
import os
from glob import glob
from utils import normalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_corpus_directory', type=str, default='./corpus0', help='json file directory')
    parser.add_argument('--normalized_corpus_directory', type=str, default='./corpus0_norm/', help='corpus directory')
    
    args = parser.parse_args()
    raw_corpus_directory = args.raw_corpus_directory
    normalized_corpus_directory = args.normalized_corpus_directory
    
    if not os.path.exists(normalized_corpus_directory):
        os.makedirs(normalized_corpus_directory)
        print('created directory {}'.format(normalized_corpus_directory))
    
    raw_corpus_fnames = glob('{}/*.txt'.format(raw_corpus_directory))
    for fname in sorted(raw_corpus_fnames):
        print('raw: {}'.format(fname))
    
    n_corpus = len(raw_corpus_fnames)
    for i_corpus, raw_corpus_fname in enumerate(sorted(raw_corpus_fnames)):
        name = raw_corpus_fname.split('/')[-1]
        normed_corpus_fname = '{}/{}'.format(normalized_corpus_directory, name)
        with open(raw_corpus_fname, encoding='utf-8') as fi:
            with open(normed_corpus_fname, 'w', encoding='utf-8') as fo:
                for i_doc, doc in enumerate(fi):
                    doc = normalize(doc)
                    fo.write('{}\n'.format(doc))
                    if i_doc % 100 == 99:
                        print('\rnormalizing ... {}/{} corpus, {} docs'.format(i_corpus+1, n_corpus, i_doc), flush=True, end='')
        print('\rnormalizing done. {}/{}, complated {} docs{}'.format(i_corpus+1, n_corpus, i_doc, ' '*30), flush=True)

if __name__ == "__main__":
    main()