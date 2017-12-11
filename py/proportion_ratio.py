def compute_a_proportion_ratio(x, seed_nodes, context_nodes, minimum_proportion_ratio=0.7, minimum_cooccurrence_frequency=10, debug=False):
    
    context_nodes = set(context_nodes)
    n = len(seed_nodes)
    
    W_ij = {}

    global_frequency = x.sum(axis=0).tolist()[0]
    global_sum = sum(global_frequency)
    
    for n_w_i, w_i in enumerate(seed_nodes):
        if debug and n_w_i >= 30:
            break
            
        positive_frequency = x[x[:,w_i].nonzero()[0],:].sum(axis=0)
        pos_sum = positive_frequency.sum()
        neg_sum = global_sum - pos_sum

        scores = []
        for w_j in positive_frequency.nonzero()[1]:
            if (w_i == w_j) or not (w_j in context_nodes):
                continue
                
            freq_pos = positive_frequency[0,w_j]
            if freq_pos < minimum_cooccurrence_frequency:
                continue

            freq_glo = global_frequency[w_j]
            freq_neg = freq_glo - freq_pos
            pos_p = freq_pos / pos_sum
            neg_p = freq_neg / neg_sum
            
            proportion_ratio = pos_p / (pos_p + neg_p)
            if proportion_ratio < minimum_proportion_ratio:
                continue
            
            scores.append((w_j, proportion_ratio, freq_pos))

        if not scores:
            continue
        W_ij[w_i] = tuple(sorted(scores, key=lambda x:-x[1]))

        if n_w_i % 10 == 9:
            print('\r{} % ({} in {})'.format('%.2f'%(100*(n_w_i+1)/n), n_w_i, n), end='', flush=True)
    print('\rconstructing was done.', flush=True)
    return W_ij

def row_labeling(x_frequency, minimum_proportion_ratio=0.7, minimum_frequency=10, verbose=False):
    global_frequency = x_frequency.sum(axis=0).tolist()[0]
    global_sum = sum(global_frequency)
    
    W_ij = {}
    
    for i, positive_frequency in _as_dict(x_frequency).items():
        pos_sum = sum(positive_frequency.values())
        neg_sum = global_sum - pos_sum
        
        scores = []
        for j, freq_pos in positive_frequency.items():
            if freq_pos < minimum_frequency:
                continue

            freq_neg = global_frequency[j] - freq_pos
            pos_p = freq_pos / pos_sum
            neg_p = freq_neg / neg_sum
            
            proportion_ratio = pos_p / (pos_p + neg_p)
            if proportion_ratio < minimum_proportion_ratio:
                continue
            
            scores.append((j, proportion_ratio, freq_pos))

        if not scores:
            continue
        W_ij[i] = tuple(sorted(scores, key=lambda x:-x[1]))

        if verbose:
            print('\r{} % ({} in {})'.format('%.2f'%(100*(n_w_i+1)/n), n_w_i, n), flush=True, end='')
    
    print('\rconstructing was done.', flush=True)
    return W_ij

def _as_dict(x):
    from collections import defaultdict
    rows, cols = x.nonzero()
    data = x.data
    dd = defaultdict(lambda: {})
    for i, j, d in zip(rows, cols, data):
        dd[i][j] = d
    return dict(dd)