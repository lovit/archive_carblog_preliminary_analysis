import argparse
import os
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
import warnings


def main():
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str, default='./base_model/', help='json file directory')
    parser.add_argument('--corpus_directory', type=str, default='./', help='corpus directory')
    parser.add_argument('--debug', dest='DEBUG', action='store_true')
    
    parser.add_argument('--do_tokenize', dest='TOKENIZE', action='store_true')
    parser.add_argument('--tokenized_corpus_directory', type=str, default='./', help='tokenized corpus directory')
    parser.add_argument('--tokenizer_name', type=str, default='twitter', help='tokenizer name', choices=['twitter'])
    parser.add_argument('--mm_file_header', type=str, default='base', help='file header of mm. eg) base_c3.mtx')
    parser.add_argument('--do_build_mm', dest='BUILD_MM', action='store_true')
    parser.add_argument('--do_merge_mm', dest='MERGE_MM', action='store_true')
    parser.add_argument('--min_tf', type=int, default=50, help='minimum term frequency for each category')
    
    parser.add_argument('--do_indi_kmeans', dest='KMEANS_INDI', action='store_true', help='category individual clustering')
    parser.add_argument('--do_whole_kmeans', dest='KMEANS_WHOLE', action='store_true', help='category individual clustering')
    parser.add_argument('--kmeans_n_jobs', type=int, default=4, help='minimum term frequency for each category')
    parser.add_argument('--k_array', type=str, default='2_5_10_20_50_100', help='k values 2_5_10 format')

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
    min_tf = args.min_tf
    BUILD_MM = args.BUILD_MM
    MERGE_MM = args.MERGE_MM
    
    KMEANS_INDI = args.KMEANS_INDI
    KMEANS_WHOLE = args.KMEANS_WHOLE
    kmeans_n_jobs = args.kmeans_n_jobs
    k_array = [int(k) for k in args.k_array.split('_')]
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
            print('Do kmeans with category = {} term frequency matrix'.format(c))
            do_kmeans(mm_indi_fname, k_array, kmeans_n_jobs, DEBUG)
    
    # Merge corpus
    if KMEANS_WHOLE:
        mm_whole_fname = '{}/{}_whole.mtx'.format(model_directory, mm_file_header)
        if not os.path.exists(mm_whole_fname):
            if not os.path.exists('{}/{}_c0.mtx'.format(mm_file_header, model_directory)):
                print('Matrix market file of individual category does not exist\nTerminate process')
                return None
            merge_mm(model_directory, num_categories, mm_file_header)
        print('Do kmeans with merged term frequency matrix')
        do_kmeans(mm_whole_fname, k_array, kmeans_n_jobs, DEBUG)
    
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

def do_kmeans(mm_fname, k_array, kmeans_n_jobs, DEBUG):
    def _do_kmeans(x, k):
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=15, n_jobs=kmeans_n_jobs)
        return kmeans.fit_predict(x)
    def _write_result(fname, labels):
        with open(fname, 'w', encoding='utf-8') as fo:
            for label in labels:
                fo.write('{}\n'.format(label))

    model_directory = '/'.join(mm_fname.split('/')[:-1])
    mm_name = mm_fname.split('/')[-1][:-4]
    # TF    
    x = mmread(mm_fname)
    for i_k, k in enumerate(k_array):
        if DEBUG and i_k == 3:
            break
        print('  - k-means (tf) begin k={} ... '.format(k), flush=True, end='')
        labels = _do_kmeans(x, k)
        labels_fname = '{}/cluster_label_tf_{}_k{}.txt'.format(model_directory, mm_name, k)
        _write_result(labels_fname, labels)
        print('done, mem={} Gb'.format('%.2f'%get_process_memory()), flush=True)
    
    # TFIDF 
    transformer = TfidfTransformer()
    x = transformer.fit_transform(x)
    for i_k, k in enumerate(k_array):
        if DEBUG and i_k == 3:
            break
        print('  - k-means (tf-idf) begin k={} ... '.format(k), flush=True, end='')
        labels = _do_kmeans(x, k)
        labels_fname = '{}/cluster_label_tfidf_{}_k{}.txt'.format(model_directory, mm_name, k)
        _write_result(labels_fname, labels)
        print('done, mem={} Gb'.format('%.2f'%get_process_memory()), flush=True)

if __name__ == '__main__':
    main()