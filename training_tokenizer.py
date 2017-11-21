import argparse
from glob import glob

from tokenizers import BranchingEntropyDictionaryBuilder
from tokenizers import DroprateScoreDictionaryBuilder
from tokenizers import WordPieceModelBuilder
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
                        default='./noname',
                        help='tokenizer_name'
                       )
    parser.add_argument('--min_frequency',
                        type=int, default=100,
                        help='minimum frequency for universial vocabulary construction'
                       )
    parser.add_argument('--subword_max_length',
                        type=int,
                        default=8,
                        help='maximum length of left-side subsection (subword)'
                       )
    parser.add_argument('--minimum_droprate_score',
                        type=float,
                        default=0.4,
                        help='minimum #(w[:-1]) / #(w)'
                       )
    parser.add_argument('--minimum_branching_entropy',
                        type=float,
                        default=1.5,
                        help='entropy of (A? | A)'
                       )
    parser.add_argument('--num_units_of_wpm',
                        type=int,
                        default=5000,
                        help='number of Word Piece Model units'
                       )
    
    args = parser.parse_args()
    
    corpus_fnames = glob('{}/*.txt'.format(args.corpus_directory))    
    tokenizer_type = args.tokenizer_type
    tokenizer_fname = args.tokenizer_fname
    min_frequency = args.min_frequency
    subword_max_length = args.subword_max_length
    minimum_droprate_score = args.minimum_droprate_score
    minimum_branching_entropy = args.minimum_branching_entropy
    num_units_of_wpm = args.num_units_of_wpm
                         
    print('{} corpus exist'.format(len(corpus_fnames)))
    for corpus_fname in corpus_fnames:
        print(corpus_fname)
    
    check_dirs(tokenizer_fname)
    
    if tokenizer_type == 'droprate':
        print('Training droprate score dictionary')
        builder = DroprateScoreDictionaryBuilder(corpus_fnames,
                                                 tokenizer_fname,
                                                 min_frequency,
                                                 subword_max_length,
                                                 minimum_droprate_score
                                                )
    elif tokenizer_type == 'branching_entropy':
        print('Training branching_entropy dictionary')
        builder = BranchingEntropyDictionaryBuilder(corpus_fnames,
                                                    tokenizer_fname,
                                                    min_frequency,
                                                    subword_max_length,
                                                    minimum_branching_entropy
                                                   )
    elif tokenizer_type == 'wpm':
        print('Training word piece model units')
        builder = WordPieceModelBuilder(corpus_fnames,
                                        subword_max_length,
                                        tokenizer_fname,
                                        num_units_of_wpm
                                       )
    
if __name__ == "__main__":
    main()