import argparse
import pickle
from scipy.io import mmread
from proportion_ratio import compute_a_proportion_ratio
from subword_df_handler import SubwordDocumentFrequencyHandler
from subword_df_handler import extract_positive_words
from subword_df_handler import extract_category_sensitive_words
from subword_df_handler import create_subword_frequency_matrix
from utils.utils import read_list
from utils.utils import remove_alphabet_number_comb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str, default='./', help='model (extraction results) directory')
    parser.add_argument('--reference_parameter_fname', type=str, default='./r.pkl', help='pickle file having index2subword and subword_df_slot')

    parser.add_argument('--pos_max_df_nstd_mean', type=float, default=1.0, help='parameter for positive word extraction')
    parser.add_argument('--pos_min_df_mean', type=float, default=0.005, help='parameter for positive word extraction')
    parser.add_argument('--ref_max_df_nstd_mean', type=float, default=1.5, help='parameter for positive word extraction')
    parser.add_argument('--ref_min_df_mean', type=float, default=0.002, help='parameter for positive word extraction')
    
    parser.add_argument('--sensitive_min_nstd', type=float, default=1.5, help='parameter for sensitive word extraction')
    parser.add_argument('--sensitive_max_mean', type=float, default=0.01, help='parameter for sensitive word extraction')
    parser.add_argument('--sensitive_min_ratio', type=float, default=2.0, help='parameter for sensitive word extraction')

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--do_tokenize', dest='TOKENIZE', action='store_true')
    parser.add_argument('--corpus_directory', type=str, default='./', help='corpus directory')
    parser.add_argument('--do_construct_graph', dest='CONSTRUCT_GRAPH', action='store_true')
    parser.add_argument('--minimum_proportion_ratio', type=float, default=0.70)
    parser.add_argument('--minimum_cooccurrence_frequency', type=int, default=10)
    
    ####################
    #### Parameters ####
    args = parser.parse_args()
    reference_parameter_fname = args.reference_parameter_fname
    model_directory = args.model_directory
    positive_parameter_fname = '{}/subword_df_slot.pkl'.format(model_directory)
    # Parameter for step 1
    pos_max_df_nstd_mean = args.pos_max_df_nstd_mean
    pos_min_df_mean = args.pos_min_df_mean
    ref_max_df_nstd_mean = args.ref_max_df_nstd_mean
    ref_min_df_mean = args.ref_min_df_mean
    # Parameter for step 2
    sensitive_min_nstd = args.sensitive_min_nstd
    sensitive_max_mean = args.sensitive_max_mean
    sensitive_min_ratio = args.sensitive_min_ratio
    # Parameter for step 3
    debug = args.debug
    TOKENIZE = args.TOKENIZE
    corpus_directory = args.corpus_directory
    CONSTRUCT_GRAPH = args.CONSTRUCT_GRAPH    
    minimum_proportion_ratio = args.minimum_proportion_ratio
    minimum_cooccurrence_frequency = args.minimum_cooccurrence_frequency
    
    print('{}\nArguments'.format('#'*80))
    args = vars(args)
    for field, value in sorted(args.items()):
        print('  -- {} = {}'.format(field, value))
    ####################
    
    # Step 1
    print('{}\nStep 1. Extract positive subwords'.format('#'*80))
    pos_subword_handler = SubwordDocumentFrequencyHandler(positive_parameter_fname)
    pos_statistics = pos_subword_handler.get_df_distribution_statistics_of_all_words()

    ref_subword_handler = SubwordDocumentFrequencyHandler(reference_parameter_fname)
    ref_statistics = ref_subword_handler.get_df_distribution_statistics_of_all_words()
    pos_positive, ref_positive, pos_filtered = extract_positive_words(pos_statistics, ref_statistics, pos_max_df_nstd_mean, pos_min_df_mean, ref_max_df_nstd_mean, ref_min_df_mean)
    
    print('  - num reference subwords  = {}'.format(len(ref_positive)))
    print('  - num positive candidates = {}'.format(len(pos_positive)))
    print('  - num positive subwords   = {}'.format(len(pos_filtered)))

    # Step 2
    print('{}\nStep 2. Extract category - sensitive subwords'.format('#'*80))
    category_sensitive_words = extract_category_sensitive_words(pos_subword_handler, pos_statistics, sensitive_min_nstd, sensitive_max_mean, sensitive_min_ratio)
    
    with open('{}/common_words.pkl'.format(model_directory), 'wb') as f:
        pickle.dump(pos_positive,f)    
    with open('{}/category_sensitive_words.pkl'.format(model_directory), 'wb') as f:
        pickle.dump(category_sensitive_words,f)
        
    print('  - num category sensitive words')
    for c, sensitive_words in enumerate(category_sensitive_words):
        print('    - category {} has {} sensitives'.format(c, len(sensitive_words)))
    
    # Step 3 - 0: create subword frequency matrix
    if TOKENIZE:
        print('{}\nStep 3 (0). Create subword frequency matrix'.format('#'*80))
        create_subword_frequency_matrix(category_sensitive_words,
                                        pos_filtered,
                                        pos_subword_handler.subword2index,
                                        corpus_directory,
                                        model_directory,
                                        debug
                                       )
    
    # Step 3 - 1: create related subwords graph
    if CONSTRUCT_GRAPH:
        print('{}\nStep 3 (1). Create related subword graph'.format('#'*80))
        context_nodes = pos_filtered
        for c in range(pos_subword_handler.num_categories):
            if debug and c >= 3:
                break

            seed_nodes = category_sensitive_words[c]
            c_context_nodes = {w for w in context_nodes}
            c_context_nodes.update(seed_nodes)

            seed_nodes = remove_alphabet_number_comb(seed_nodes)
            seed_nodes = {pos_subword_handler.encode(w) for w in seed_nodes}
            c_context_nodes = remove_alphabet_number_comb(c_context_nodes)
            c_context_nodes = {pos_subword_handler.encode(w) for w in c_context_nodes}

            x_fname = '{}/positive_subword_tf_c{}.mtx'.format(model_directory, c)
            x = mmread(x_fname).tocsr()
            W_ij = compute_a_proportion_ratio(x, seed_nodes, c_context_nodes, minimum_proportion_ratio, minimum_cooccurrence_frequency, debug)
            W_ij = {pos_subword_handler.decode(w_i):tuple([(pos_subword_handler.decode(w_j), score, freq) for w_j, score, freq in w_js]) for w_i, w_js in W_ij.items()}

            graph_fname = '{}/related_words_c{}.pkl'.format(model_directory, c)
            with open(graph_fname, 'wb') as f:
                pickle.dump(W_ij, f)
            print('proportion ratio was saved, category = {}\n'.format(c))


if __name__ == "__main__":
    main()