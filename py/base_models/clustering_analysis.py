import argparse
import os
import pickle
from collections import defaultdict
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import warnings
import numpy as np

def main():
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str, default='./base_model/', help='json file directory')
    parser.add_argument('--debug', dest='DEBUG', action='store_true')
    parser.add_argument('--mm_file_header', type=str, default='base', help='file header of mm. eg) base_c3.mtx')
    parser.add_argument('--k_array', type=str, default='2_5_10_20_50_100', help='k values 2_5_10 format')
    parser.add_argument('--num_categories', type=int, default=27, help='number of categories')
    
    parser.add_argument('--do_indi_analysis', dest='INDI_ANALYSIS', action='store_true')
    parser.add_argument('--do_whole_analysis', dest='WHOLE_ANALYSIS', action='store_true')

    ###################
    #### PARAMETER ####
    args = parser.parse_args()
    model_directory = args.model_directory    
    mm_file_header = args.mm_file_header
    num_categories = args.num_categories    
    k_array = [int(k) for k in args.k_array.split('_')]
    DEBUG = args.DEBUG
    INDI_ANALYSIS = args.INDI_ANALYSIS
    WHOLE_ANALYSIS = args.WHOLE_ANALYSIS
    ###################
    
    print('{}\nArguments'.format('#'*80))
    args = vars(args)
    for field, value in sorted(args.items()):
        print('  -- {} = {}'.format(field, value))
    ###################
    ###################
    
    if INDI_ANALYSIS:
        weight_types = ['tf', 'tfidf']
        for weight_type in weight_types:
            for k in k_array:
                centroid_array = []
                centroid_norm_array = []
                num_rows_array = []
                print('  - k={}, weight={} making centroid vector from all categories'.format(k, weight_type))
                for c in range(num_categories):
                    print('\r    - loading category = {} / {} ...'.format(c, num_categories), flush=True, end='')
                    mm_fname = '{}/{}_c{}.mtx'.format(model_directory, mm_file_header, c)
                    cluster_label_fname = '{}/cluster_label_{}_{}_c{}_k{}.txt'.format(model_directory, weight_type, mm_file_header, c, k)
                    cent, cent_norm, rows = make_centroid_vector(mm_fname, cluster_label_fname)
                    for cent_i, cent_norm_i, rows_i in zip(cent, cent_norm, rows):
                        centroid_array.append(cent_i)
                        centroid_norm_array.append(cent_norm_i)
                        num_rows_array.append((c, rows_i))
                tf = np.concatenate(centroid_array)
                tf_norm = np.concatenate(centroid_norm_array)
                centroid_vectors_fname = '{}/centroids_{}_{}_k{}.pkl'.format(model_directory, weight_type, mm_file_header, k)
                with open(centroid_vectors_fname, 'wb') as f:
                    params = {'tf':tf, 'tf_norm':tf_norm, 'n_rows':num_rows_array}
                    pickle.dump(params, f)
                print('\r  - k={}, weight={} centroids from all categoreis was pickled'.format(k, weight_type))
    
def make_centroid_vector(mm_fname, cluster_label_fname):
    x = mmread(mm_fname).tocsr()
    labels = _load_list(cluster_label_fname)
    groupby = defaultdict(lambda: [])
    for i, group in enumerate(labels):
        groupby[group].append(i)
    
    centroid_array = []
    centroid_norm_array = []
    rows_array = []
    
    for group, rows in sorted(groupby.items()):
        x_sub = x[rows,:]
        centroid = x.sum(axis=0)
        centroid_norm = normalize(centroid)
        
        centroid_array.append(centroid)
        centroid_norm_array.append(centroid_norm)
        rows_array.append(rows)
    return centroid_array, centroid_norm_array, rows_array

def _load_list(fname):
    with open(fname, encoding='utf-8') as f:
        docs = [int(doc.strip()) for doc in f]
    return docs
        
if __name__ == '__main__':
    main()