import argparse
import os
import pickle
from collections import Counter
from collections import defaultdict
from glob import glob
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from konlpy.tag import Twitter
from utils.utils import get_process_memory
import numpy as np

try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    print('Package warning does not exist')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str, default='./base_model/', help='json file directory')
    parser.add_argument('--corpus_directory', type=str, default='./', help='corpus directory')
    parser.add_argument('--debug', dest='DEBUG', action='store_true')
    
    parser.add_argument('--do_tokenize', dest='TOKENIZE', action='store_true')
    parser.add_argument('--tokenized_corpus_directory', type=str, default='./', help='tokenized corpus directory')
    parser.add_argument('--tokenizer_name', type=str, default='twitter', help='tokenizer name', choices=['twitter'])
    parser.add_argument('--mm_file_header', type=str, default='base', help='file header of mm. eg) base_c3.mtx')
    parser.add_argument('--output_header', type=str, default='', help='file header of output')
    parser.add_argument('--do_build_mm', dest='BUILD_MM', action='store_true')
    parser.add_argument('--do_merge_mm', dest='MERGE_MM', action='store_true')
    parser.add_argument('--min_tf', type=int, default=50, help='minimum term frequency for each category')
    
    parser.add_argument('--do_indi_kmeans', dest='KMEANS_INDI', action='store_true', help='category individual clustering')
    parser.add_argument('--do_whole_kmeans', dest='KMEANS_WHOLE', action='store_true', help='category individual clustering')
    parser.add_argument('--kmeans_n_jobs', type=int, default=4, help='minimum term frequency for each category')
    parser.add_argument('--k_array', type=str, default='2_5_10_20_50_100', help='k values 2_5_10 format')

    parser.add_argument('--do_indi_analysis', dest='INDI_ANALYSIS', action='store_true')
    parser.add_argument('--do_whole_analysis', dest='WHOLE_ANALYSIS', action='store_true')
    parser.add_argument('--proportion_minimum_df_ratio', type=float, default=0.03, help='minimum document frequency ratio for term proportion vector')
    
    ###################
    #### PARAMETER ####
    args = parser.parse_args()
    model_directory = args.model_directory
    corpus_directory = args.corpus_directory
    DEBUG = args.DEBUG
    
    TOKENIZE = args.TOKENIZE
    tokenized_corpus_directory = args.tokenized_corpus_directory
    tokenizer_name = args.tokenizer_name
    mm_file_header = args.mm_file_header
    output_header = args.output_header
    min_tf = args.min_tf
    BUILD_MM = args.BUILD_MM
    MERGE_MM = args.MERGE_MM
    
    KMEANS_INDI = args.KMEANS_INDI
    KMEANS_WHOLE = args.KMEANS_WHOLE
    kmeans_n_jobs = args.kmeans_n_jobs
    k_array = [int(k) for k in args.k_array.split('_')]

    INDI_ANALYSIS = args.INDI_ANALYSIS
    WHOLE_ANALYSIS = args.WHOLE_ANALYSIS
    proportion_minimum_df_ratio = args.proportion_minimum_df_ratio
    ###################
    
    print('{}\nArguments'.format('#'*80))
    args = vars(args)
    for field, value in sorted(args.items()):
        print('  -- {} = {}'.format(field, value))
    ###################
    ###################
    
    num_categories = len(glob('{}/*.txt'.format(corpus_directory)))
    
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # Tokenizing
    if TOKENIZE:
        tokenizer = Twitter()
        if not os.path.exists(tokenized_corpus_directory):
            os.makedirs(tokenized_corpus_directory)
        tokenize(corpus_directory, tokenized_corpus_directory, tokenizer)

    if BUILD_MM:
        build_mm(tokenized_corpus_directory, model_directory, min_tf, mm_file_header)

    if not BUILD_MM and MERGE_MM:
        merge_mm(model_directory, num_categories, mm_file_header)
    
    if KMEANS_INDI:
        for c in range(num_categories):
            if DEBUG and c == 3:
                break
            mm_indi_fname = '{}/{}_c{}.mtx'.format(model_directory, mm_file_header, c)
            print('Do kmeans with category = {}'.format(c))
            do_kmeans(mm_indi_fname, k_array, kmeans_n_jobs, DEBUG, output_header)
    
    # Merge corpus
    if KMEANS_WHOLE:
        mm_whole_fname = '{}/{}_whole.mtx'.format(model_directory, mm_file_header)
        if not os.path.exists(mm_whole_fname):
            if not os.path.exists('{}/{}_c0.mtx'.format(mm_file_header, model_directory)):
                print('Matrix market file of individual category does not exist\nTerminate process')
                return None
            merge_mm(model_directory, num_categories, mm_file_header)
        print('Do kmeans with merged x')
        do_kmeans(mm_whole_fname, k_array, kmeans_n_jobs, DEBUG, output_header)
    
    # make proportion
    if INDI_ANALYSIS:
        for weight_type in ['tf', 'tfidf']:
            print('Do clustering result (indi) analysis {}'.format(weight_type))
            indi_analysis(model_directory, mm_file_header, k_array, num_categories, weight_type, proportion_minimum_df_ratio, DEBUG, output_header)
    
def tokenize(corpus_directory, tokenized_corpus_directory, tokenizer):
    def normalize(doc):
        import re
        only_hangle = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]')
        doublespace = re.compile('\s+')
        return doublespace.sub(' ', only_hangle.sub(' ', doc))
    
    print('{}\nTokenization begin'.format('#'*80))
    corpus_fnames = sorted(glob('{}/*.txt'.format(corpus_directory)))
    for corpus_fname in corpus_fnames:
        print('  - corpus: {}'.format(corpus_fname))
    print()
    
    n_corpus = len(corpus_fnames)
    for i_corpus, corpus_fname in enumerate(corpus_fnames):
        corpus_name = corpus_fname.split('/')[-1]
        tokenized_corpus_fname = '{}/{}'.format(tokenized_corpus_directory, corpus_name)
        with open(corpus_fname, encoding='utf-8') as fi:            
            with open(tokenized_corpus_fname, 'w', encoding='utf-8') as fo:
                for i_doc, doc in enumerate(fi):
                    if i_doc % 100 == 99:
                        print('\r  - tokenizing {} / {} corpus, {} docs'.format(i_corpus+1, n_corpus, i_doc+1), flush=True, end='')
                    doc = normalize(doc)
                    tokens = tokenizer.pos(doc.strip())
                    tokens = ['{}/{}'.format(w,t) for w,t in tokens]
                    doc_ = ' '.join(tokens)
                    fo.write('{}\n'.format(doc_))
        print('\rtokenizing {} / {} was done. n_docs = {}{}'.format(i_corpus+1, n_corpus, i_doc+1, ' '*30, flush=True))
    print('All corpus was tokenized\n')

def build_mm(tokenized_corpus_directory, model_directory, min_tf, mm_file_header):
    corpus_fnames = sorted(glob('{}/*.txt'.format(tokenized_corpus_directory)))
    universial_vocabulary = {}
    n_corpus = len(corpus_fnames)
    
    print('{}\nBuilding mm begin'.format('#'*80))
    
    # Scanning vocabulary in all corpus
    for i_corpus, corpus_fname in enumerate(corpus_fnames):
        # Scanning vocabulary for pruning
        vocabulary = defaultdict(int)
        with open(corpus_fname, encoding='utf-8') as f:
            for i_doc, doc in enumerate(f):
                if i_doc % 1000 == 999:
                    print('\r  - scanning vocabulary {} / {} corpus, {} docs'.format(i_corpus+1, n_corpus, i_doc+1), flush=True, end='')
                for word in doc.split():
                    vocabulary[word] += 1
        for vocab, count in vocabulary.items():
            if count < min_tf:
                continue
            if not (vocab in universial_vocabulary):
                universial_vocabulary[vocab] = len(universial_vocabulary)
        print('\rscanning vocabulry in {} / {} corpus was done mem={} Gb{}'.format(i_corpus+1, n_corpus, '%.2f'%get_process_memory(), ' '*40))
    
    # Save vocablary mapper
    vocab_fname = '{}/{}_vocabulary.txt'.format(model_directory, mm_file_header)
    with open(vocab_fname, 'w', encoding='utf-8') as f:
        for vocab, _ in sorted(universial_vocabulary.items(), key=lambda x:x[1]):
            f.write('{}\n'.format(vocab))
    
    # Building mm file for all categories
    n_vocabs = len(universial_vocabulary)
    for i_corpus, corpus_fname in enumerate(corpus_fnames):
        # Create mm file for each category
        rows = []
        cols = []
        data = []
        with open(corpus_fname, encoding='utf-8') as f:
            for i_doc, doc in enumerate(f):
                if i_doc % 1000 == 999:
                    print('\r  - building matrix market {} / {} corpus, {} docs'.format(i_corpus+1, n_corpus, i_doc+1), flush=True, end='')
                words = [universial_vocabulary.get(word,-1) for word in doc.split()]
                words = Counter([word for word in words if word >= 0])
                for j_word, count in words.items():
                    rows.append(i_doc)
                    cols.append(j_word)
                    data.append(count)
        mm_name = corpus_fname.split('/')[-1].replace('.txt', '.mtx')
        mm_fname = '{}/{}_c{}'.format(model_directory, mm_file_header, mm_name)
        x = csr_matrix((data, (rows, cols)), shape=(i_doc+1, n_vocabs))
        mmwrite(mm_fname, x)
        print('\rmatrix market of {} / {} corpus was saved. mem={} Gb, shape={}'.format(i_corpus+1, n_corpus, '%.2f'%get_process_memory(), x.shape), flush=True)
        del x
    print('ALl corpus were transformed into matrix market files')
    merge_mm(model_directory, n_corpus, mm_file_header)

def merge_mm(model_directory, num_categories, mm_file_header):
    print('{}\nMerging mm begin'.format('#'*80))
    
    whole_mm_fname = '{}/{}_whole.mtx'.format(model_directory, mm_file_header)
    # Get total document number
    n_docs = 0
    n_vocabs = 0
    n_elements = 0
    for c in range(num_categories):
        indi_mm_fname = '{}/{}_c{}.mtx'.format(model_directory, mm_file_header, c)
        print('  -- to be merged: {}'.format(indi_mm_fname))
        if c >= 3:
            print('  ...')
            break
    
    for c in range(num_categories):
        indi_mm_fname = '{}/{}_c{}.mtx'.format(model_directory, mm_file_header, c)
        with open(indi_mm_fname, encoding='utf-8') as f:
            header = next(f)
            second = next(f)
            doc = next(f).strip()
            n_docs += int(doc.split()[0])
            n_vocabs = int(doc.split()[1])
            n_elements += int(doc.split()[2])
    # Merge
    with open(whole_mm_fname, 'w', encoding='utf-8') as fo:
        fo.write(header)
        fo.write(second)
        fo.write('{} {} {}\n'.format(n_docs, n_vocabs, n_elements))
        base_n_docs = 0
        for c in range(num_categories):
            indi_mm_fname = '{}/{}_c{}.mtx'.format(model_directory, mm_file_header, c)
            with open(indi_mm_fname, encoding='utf-8') as fi:
                for _ in range(2):
                    next(fi)
                n_docs = int(next(fi).split()[0])
                for doc in fi:
                    i, j, v = doc.split()
                    fo.write('{} {} {}\n'.format(int(i)+base_n_docs, j, v))
            base_n_docs += n_docs
            print('  - merged {}'.format(indi_mm_fname))
    print('All corpus was merged')

def do_kmeans(mm_fname, k_array, kmeans_n_jobs, DEBUG, output_header):
    def _do_kmeans(x, k):
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=15, n_jobs=kmeans_n_jobs)
        labels = kmeans.fit_predict(x)
        centers = kmeans.cluster_centers_
        return labels, centers
    def _write_labels(fname, labels):
        with open(fname, 'w', encoding='utf-8') as fo:
            for label in labels:
                fo.write('{}\n'.format(label))
    def _write_centers(fname, centers):
        with open(fname, 'wb') as fo:
            pickle.dump(centers, fo)

    model_directory = '/'.join(mm_fname.split('/')[:-1])
    mm_name = mm_fname.split('/')[-1][:-4]
    # TF    
    x = mmread(mm_fname)
    if output_header:
        output_header += '_'
        
    for i_k, k in enumerate(k_array):
        if DEBUG and i_k == 3:
            break
        print('  - k-means (tf) begin k={} ... '.format(k), flush=True, end='')
        labels, centers = _do_kmeans(x, k)
        labels_fname = '{}/cluster_label_tf_{}_{}k{}.txt'.format(model_directory, mm_name, output_header, k)
        _write_labels(labels_fname, labels)
        centers_fname = '{}/cluster_center_tf_{}_{}k{}.pkl'.format(model_directory, mm_name, output_header, k)
        _write_centers(centers_fname, centers)
        print('done, mem={} Gb'.format('%.2f'%get_process_memory()), flush=True)
    
    # TFIDF 
    transformer = TfidfTransformer()
    x = transformer.fit_transform(x)
    for i_k, k in enumerate(k_array):
        if DEBUG and i_k == 3:
            break
        print('  - k-means (tf-idf) begin k={} ... '.format(k), flush=True, end='')
        labels, centers = _do_kmeans(x, k)
        labels_fname = '{}/cluster_label_tfidf_{}_{}k{}.txt'.format(model_directory, mm_name, output_header, k)
        _write_labels(labels_fname, labels)
        centers_fname = '{}/cluster_center_tfidf_{}_{}k{}.pkl'.format(model_directory, mm_name, output_header, k)
        _write_centers(centers_fname, centers)
        print('done, mem={} Gb'.format('%.2f'%get_process_memory()), flush=True)

def indi_analysis(model_directory, mm_file_header, k_array, num_categories, weight_type, proportion_minimum_df_ratio=0.03, DEBUG=False, output_header=''):
    group_by_k = {}
    tf_center_by_k = defaultdict(lambda: [])
    tfidf_center_by_k = defaultdict(lambda: [])
    
    if output_header:
        output_header += '_'
    
    for c in range(num_categories):
        if DEBUG and c >= 3:
            break
        print('\r  - clustering analysis = {} / {} categories begin ...{}'.format(c, num_categories, ' '*20), flush=True)
        mm_fname = '{}/{}_c{}.mtx'.format(model_directory, mm_file_header, c)
        x = mmread(mm_fname).tocsr()
        n_vocabs = x.shape[1]
        
        for i_k, k in enumerate(k_array):
            if DEBUG and i_k == 3:
                break
            print('\r    - analyzing k = {} in {}'.format(k, k_array), flush=True, end='')
            cluster_label_fname = '{}/cluster_label_{}_{}_c{}_{}k{}.txt'.format(model_directory, weight_type, mm_file_header, c, output_header, k)
            labels = _load_list(cluster_label_fname)
            
            proportions, dfs, n_docs = group_by_k.get(k, ([], [], []))
            proportions_k, dfs_k, n_docs_k = _make_proportion(x, k, labels, proportion_minimum_df_ratio)
            for p_i, df_i, nd_i in zip(proportions_k, dfs_k, n_docs_k):
                proportions.append(p_i)
                dfs.append(df_i)
                n_docs.append(nd_i)
            group_by_k[k] = (proportions, dfs, n_docs)
            
            centers_fname = '{}/cluster_center_tf_{}_c{}_{}k{}.pkl'.format(model_directory, mm_file_header, c, output_header, k)
            with open(centers_fname, 'rb') as f:
                tf_center_by_k[k].append(pickle.load(f))
                
            centers_fname = '{}/cluster_center_tfidf_{}_c{}_{}k{}.pkl'.format(model_directory, mm_file_header, c, output_header, k)
            with open(centers_fname, 'rb') as f:
                tfidf_center_by_k[k].append(pickle.load(f))
    
    print('\r  - pickling ... {}'.format(' '*40), flush=True, end='')
    for k, args in group_by_k.items():
        proportion_fname = '{}/proportions_{}_{}_{}k{}.pkl'.format(model_directory, weight_type, mm_file_header, output_header, k)
        _packing(*args, n_vocabs, proportion_fname)
    for k, center_list in tf_center_by_k.items():
        x = np.concatenate(center_list)
        centers_fname = '{}/cluster_center_tf_{}_{}k{}.pkl'.format(model_directory, mm_file_header, output_header, k)
        with open(centers_fname, 'wb') as f:
            pickle.dump(x, f)
    for k, center_list in tfidf_center_by_k.items():
        x = np.concatenate(center_list)
        centers_fname = '{}/cluster_center_tfidf_{}_{}k{}.pkl'.format(model_directory, mm_file_header, output_header, k)
        with open(centers_fname, 'wb') as f:
            pickle.dump(x, f)
    print('\rdone{}'.format(' '*40), flush=True)

def _get_rows_from_label(label, labels):
    return [i for i, label_i in enumerate(labels) if label_i == label]

def _proportion(x_sub, min_df):
    n_docs, n_vocabs = x_sub.shape
    doc_norm = {}
    word_df = {}
    # calculate norm
    rows, cols = x_sub.nonzero()
    for i, j, v in zip(rows, cols, x_sub.data):
        doc_norm[i] = doc_norm.get(i,0) + v
        word_df[j] = word_df.get(j, 0) + 1
    word_df = {j:df for j,df in word_df.items() if df >= min_df}
    proportion = {}
    # normalize
    for i, j, v in zip(rows, cols, x_sub.data):
        if not (j in word_df):
            continue
        proportion[j] = proportion.get(j,0) + (v / doc_norm[i])
    # averaging for making proportion
    proportion = {j:v/n_docs for j,v in proportion.items()}
    return proportion, word_df, n_docs

def _make_proportion(x, n_clusters, labels, proportion_minimum_df_ratio):
    proportions, dfs, n_docs = [], [], []    
    for k in range(n_clusters):        
        rows = _get_rows_from_label(k, labels)
        min_df = max(1, proportion_minimum_df_ratio * len(rows))
        x_sub = x[rows,:]
        proportion_k, df_k, n_docs_k = _proportion(x_sub, min_df)
        proportions.append(proportion_k)
        dfs.append(df_k)
        n_docs.append(n_docs_k)
    return proportions, dfs, n_docs

def _packing(proportions, dfs, n_docs, n_vocabs, fname):
    x = _list_of_d_as_sparse(proportions, n_vocabs)
    with open(fname, 'wb') as f:
        pickle.dump({'proportion':x, 'dfs': dfs, 'n_docs':n_docs}, f)
    
def _list_of_d_as_sparse(list_of_d, n_vocabs):
    rows, cols, data = [], [], []
    n_docs = len(list_of_d)
    for i, j_dict in enumerate(list_of_d):
        for j, v in j_dict.items():
            rows.append(i)
            cols.append(j)
            data.append(v)
    return csr_matrix((data, (rows, cols)),shape=(n_docs, n_vocabs))

def _load_list(fname):
    with open(fname, encoding='utf-8') as f:
        docs = [int(doc.strip()) for doc in f]
    return docs

if __name__ == '__main__':
    main()