class SubwordDocumentFrequencyHandler:
    def __init__(self, parameter_fname):
        with open(parameter_fname, 'rb') as f:
            import pickle
            params = pickle.load(f)
        self.subword_slot = params['df']
        self.index2subword = params['index2subword']
        self.subword2index = {word:index for index,word in enumerate(self.index2subword)}
        self.num_words, self.num_categories = self.subword_slot.shape
        
    def decode(self, idx):
        if 0 <= idx < self.num_words:
            return self.index2subword[idx]
        return None
    
    def encode(self, subword):
        return self.subword2index.get(subword, -1)
    
    def get_df_distribution_statistics_from_word(self, word):
        return self.get_df_distribution_statistics_from_word_index(self.encode(word))
    
    def get_df_distribution_statistics_from_word_index(self, idx):
        if idx == -1: return None
        df_dist = self.subword_slot[idx,:]
        return df_dist.std() / df_dist.mean(), df_dist.mean(), df_dist.max() / df_dist.mean(), df_dist.argmax()
    
    def get_df_distribution_statistics_of_all_words(self):
        nstd_mean = {}
        for idx in range(self.num_words):
            if idx % 1000 == 999:
                print('\r  computing (std/mean, mean, max/mean, argmax) {} / {}'.format(idx+1, self.num_words), flush=True, end='')
            nstd_mean[self.decode(idx)] = self.get_df_distribution_statistics_from_word_index(idx)
        print('\r computing (std/mean, mean, max/mean, argmax) was done.        ')
        return nstd_mean

# Step 1. Extracting positive (category-common) words
def extract_positive_words(pos_nstd_mean,
                           ref_nstd_mean,
                           pos_max_df_nstd_mean=1.0,
                           pos_min_df_mean=0.005,
                           ref_max_df_nstd_mean=1.5,
                           ref_min_df_mean=0.002):
    
    pos_positive = dict(filter(lambda x:(x[1][0]<pos_max_df_nstd_mean and x[1][1]>pos_min_df_mean), pos_nstd_mean.items()))
    ref_positive = dict(filter(lambda x:(x[1][0]<ref_max_df_nstd_mean and x[1][1]>ref_min_df_mean), ref_nstd_mean.items()))
    pos_filtered = {word:score for word, score in pos_positive.items() if not (word in ref_positive)}
    return pos_positive, ref_positive, pos_filtered

# Step 2. Extracting category-sensitive words for each category
def extract_category_sensitive_words(pos_handler, pos_statistics, min_nstd=2.5, max_mean=0.01, min_ratio=3):
    word_by_category = [[] for _ in range(pos_handler.num_categories)]
    for subword, (nstd, mean, topmean, max_sensitive_category) in pos_statistics.items():
        if nstd < min_nstd:
            continue
        if mean > max_mean:
            continue
        idx = pos_handler.encode(subword)
        df_dist = pos_handler.subword_slot[idx,:]
        sensitive_categories = [i for i, r in enumerate(df_dist/mean) if r >= min_ratio]
        for c in sensitive_categories:
            word_by_category[c].append(subword)
    return word_by_category

# Step 3 - 0. Tokenization: creating sparse matrix
from scipy.sparse import csr_matrix

def _subword_frequency_matrix(corpus_fname, subword_set, subword2index, c, n_categories, debug=False):
    def tokenize_subword(token, max_length=8):
        subwords = [token[:e] for e in range(2, min(len(token), max_length)+1)]
        subwords = [subword for subword in subwords if subword in subword_set]
        return subwords
    
    rows = []
    cols = []
    data = []
    with open(corpus_fname, encoding='utf-8') as f:
        for i, doc in enumerate(f):
            if i % 100 == 99:
                print('\r  - subword-tokenizing ... {} docs'.format(i+1), flush=True, end='')
            if debug and i >= 500:
                break
            subwords = Counter((subword for token in doc.split() for subword in tokenize_subword(token)))
            for subword, count in subwords.items():
                j = subword2index.get(subword, -1)
                if j == -1:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(count)
    (n, m) = (i+1, len(subword2index))
    x = csr_matrix((data, (rows, cols)), shape=(n,m))
    print('\rcreated subword frequency matrix in {} / {} corpus'.format(c, n_categories), flush=True)
    return x

def create_subword_frequency_matrix(sensitive_words_by_category, common_words, subword2index, corpus_directory, model_directory, debug=False):
    n_categories = len(sensitive_words_by_category)
    for c, sensitive_words in enumerate(sensitive_words_by_category):
        if debug and c >= 3: 
            break
        
        corpus_fname = '{}/{}.txt'.format(corpus_directory, c)
        
        subword_set = {word for word in common_words}
        subword_set.update({word for word in sensitive_words})
        x = _subword_frequency_matrix(corpus_fname, subword_set, subword2index, c, n_categories, debug)
        x_fname = '{}/positive_subword_tf_c{}.mtx'.format(model_directory, c)
        mmwrite(x_fname, x)
        del x
        
        subword_set = {word for word in subword2index}
        x = _subword_frequency_matrix(corpus_fname, subword_set, subword2index, c, n_categories, debug)
        x_fname = '{}/subword_tf_c{}.mtx'.format(model_directory, c)
        mmwrite(x_fname, x)
        del x

# Step 3 - 1 Select specific terms using related term graph
def select_specific_words(W_ij, sensitive_words, common_words, cost_factor=1.0, cutoff=0.05):
    def specific_score(word, cost, normalizer):
        pos, neg = (0, 0)
        related_words = W_ij.get(word, [])
        if not related_words:
            return 0, False
        for context, score, freq in related_words:
            if context in common_words:
                neg += score
            else:
                pos += score
        return normalizer * (cost * pos - neg), True
    
    cost = cost_factor * len(common_words) / (1+len(sensitive_words))
    normalizer_factor = 1 / (len(sensitive_words) + len(common_words))
    
    scored = [(word, *specific_score(word, cost, normalizer_factor)) for word in sensitive_words]
    has_no_relateds = [(t[0], 0) for t in scored if not t[2]]
    scored = [(t[0], t[1]) for t in scored if t[2]]
    true_specific = [t for t in scored if t[1] >= cutoff]
    true_specific = sorted(true_specific, key=lambda x:-x[1])
    false_specific = [t for t in scored if t[1] < cutoff]
    false_specific = sorted(false_specific, key=lambda x:-x[1])
    return true_specific, false_specific, has_no_relateds

# Step 4. Document classification
def classify_documents(x, positive_features, negative_features, negative_score_factor=1.0):
    n, m = x.shape
    factor = negative_score_factor * len(negative_features) / len(positive_features)
    
    rows, cols = x.nonzero()
    data = x.data
    scores = [0]*n
    unknown_features = [0]*n
    num_features = [0]*n
    normalizer = [0]*n
    
    n_entry = len(rows)
    for i_entry, (i,j,d) in enumerate(zip(rows, cols, data)):
        if i_entry % 5000 == 4999:
            print('\r  - computing ... {} % entries'.format('%.2f'%(100*(i_entry+1)/n_entry)), flush=True, end='')
        num_features[i] = num_features[i] + d
        if j in positive_features:
            scores[i] = scores[i] + d
            normalizer[i] = normalizer[i] + d
        if j in negative_features:
            scores[i] = scores[i] - (factor * d)
            normalizer[i] = normalizer[i] + (factor * d)
        if not (j in positive_features or j in negative_features):
            unknown_features[i] = unknown_features[i] + d
            #scores[i] = scores[i] - d * factor
    scores = [score/normalizer[i] if normalizer[i] > 0 else 0 for i,score in enumerate(scores)]
    unknown_features = [num/num_features[i] for i,num in enumerate(unknown_features) if num_features[i] > 0]
    print('\rclassification was done.{}'.format(' '*50))
    return scores, num_features, unknown_features