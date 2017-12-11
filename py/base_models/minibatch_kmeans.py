def to_minibatch_input(num_categories, directory, mm_file_header):
    # Scanning
    n_docs, n_vocabs = 0, 0
    for c in range(num_categories):
        mm_file = '{}/{}_c{}.mtx'.format(directory, mm_file_header, c)
        with open(mm_file, encoding='utf-8') as f:
            for _ in range(2):
                next(f)
            n_docs_c, n_vocabs_c, _ = next(f).split()
            n_docs += int(n_docs_c)
            n_vocabs = int(n_vocabs_c)
    # Padding document begin
    begin_docs = 0
    for c in range(num_categories):
        mm_file_in = '{}/{}_c{}.mtx'.format(directory, mm_file_header, c)
        mm_file_out = '{}/{}_minibatch_c{}.mtx'.format(directory, mm_file_header, c)

        with open(mm_file_in, encoding='utf-8') as fi:
            with open(mm_file_out, 'w', encoding='utf-8') as fo:
                for _ in range(2):
                    fo.write(next(fi))
                n_docs_c, _, n_elements = next(fi).split()
                n_elements = int(n_elements)
                fo.write('{} {} {}\n'.format(int(n_docs_c) + begin_docs, n_vocabs, n_elements))
                for i_row, row in enumerate(fi):
                    i, j, v = row.split()
                    i = int(i) + begin_docs
                    fo.write('{} {} {}\n'.format(i, j, v))
                    if i_row % 10000 == 9999:
                        print('\r  - padding {} / {} corpus, {} %'.format(c+1, num_categories, '%.2f'%(100*i_row/n_elements)), flush=True, end='')
                print('\rpadding {} / {} corpus was done.        '.format(c+1, num_categories))
        begin_docs += int(n_docs_c)