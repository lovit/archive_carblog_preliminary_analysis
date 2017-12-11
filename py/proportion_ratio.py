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