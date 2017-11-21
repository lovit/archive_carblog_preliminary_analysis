import argparse
import pickle
from glob import glob

from df_distribution import make_document_frequency_distribution
from tokenizers import SubwordMatchTokenizer
from tokenizers import WordPieceModelTokenizer
from utils import check_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_directory',
                        type=str,
                        default='./',
                        help='corpus directory'
                       )
    parser.add_argument('--tokenizer_type',
                        type=str,
                        default='droprate',
                        choices=['droprate', 'branching_entropy', 'wpm'],
                        help='tokenizer type'
                       )
    parser.add_argument('--tokenizer_fname',
                        type=str,
                        default='./tokenizer',
                        help='tokenizer model file name'
                       )
    parser.add_argument('--distribution_fname',
                        type=str,
                        default='./df.pkl',
                        help='(subword, df) numpy.array + subword index pkl file'
                       )
    
    args = parser.parse_args()
    corpus_fnames = glob('{}/*.txt'.format(args.corpus_directory))
    tokenizer_type = args.tokenizer_type
    tokenizer_fname = args.tokenizer_fname
    distribution_fname = args.distribution_fname
    
    print('{} corpus exist'.format(len(corpus_fnames)))
    for corpus_fname in corpus_fnames:
        print(corpus_fname)
    
    check_dirs(distribution_fname)
    
    if tokenizer_type == 'wpm':
        tokenizer = WordPieceModelTokenizer(tokenizer_fname)
    else:
        tokenizer = SubwordMatchTokenizer(tokenizer_fname)
    
    subword_slot, index2subword = make_document_frequency_distribution(corpus_fnames, tokenizer)
    with open(distribution_fname, 'wb') as f:
        params = {'df':subword_slot, 'index2subword':index2subword}
        pickle.dump(params, f)
    print('done')
            
if __name__ == "__main__":
    main()