# Step 2. Extracting sensitive words for each category
def extract_category_sensitive_words(pos_handler,
                                 pos_statistics,
                                 pos_min_df_nstd=2.5,
                                 pos_df_max_mean=0.01,
                                 pos_min_of_df_mean_ratio=3
                                ):
    word_by_category = [[] for _ in range(pos_handler.num_categories)]
    for subword, (nstd, mean, topmean, max_sensitive_category) in pos_statistics.items():
        if nstd < pos_min_df_nstd:
            continue
        if mean > pos_df_max_mean:
            continue
        idx = pos_handler.encode(subword)
        df_dist = pos_handler.subword_slot[idx,:]
        sensitive_categories = [i for i, r in enumerate(df_dist/mean) if r >= pos_min_of_df_mean_ratio]
        for c in sensitive_categories:
            word_by_category[c].append(subword)
    return word_by_category

# Step 3. Extracting specpfic words for each category
def does_cooccurred(w_sens, w_pos, doc):
    return (w_sens in doc) and (w_pos in doc)

def _calculate_cooccurrence_df(D_sensitive, D_positive, corpus_fname):
    cooccurrence = defaultdict(lambda: defaultdict(int))
    with open(corpus_fname, encoding='utf-8') as f:
        for i_doc, doc in enumerate(f):
            doc = ' ' + doc
            for i_ws, w_sens in enumerate(D_sensitive):
                for w_pos in D_positive:
                    if not does_cooccurred(w_sens, w_pos, doc):
                        continue
                    cooccurrence[w_sens][w_pos] += 1
            if i_doc % 10 == 9:
                print('\r  - calculating cooccurrence ... {} docs'.format(i_doc+1), flush=True, end='')
    return {w_sens:dict(cooc_dict) for w_sens, cooc_dict in cooccurrence.items()}

def calculate_cooccurrence_df(sensitive_words_by_category, D_positive, corpus_directory):
    D_positive = {' '+word for word in D_positive}
    word_by_category = []
    
    for c, D_sensitive in enumerate(sensitive_words_by_category):
        D_sensitive = {' '+word for word in D_sensitive}
        corpus_fname = '{}/{}.txt'.format(corpus_directory, c)

        process_time = time.time()
        word_by_category.append(_calculate_cooccurrence_df(D_sensitive, D_positive, corpus_fname))
        process_time = time.time() - process_time
        process_time = str(datetime.timedelta(seconds=process_time))

        print('\rcalculating cooccurrence in {} / {} corpus ({})'.format(c, len(sensitive_words_by_category), process_time), flush=True)
    return word_by_category

