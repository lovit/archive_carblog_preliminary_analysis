from utils import DoublespaceLineCorpus
from utils import get_process_memory

class DroprateScoreDictionaryBuilder:
    def __init__(self, corpus_fnames, model_fname, min_frequency, subword_max_length, minimum_droprate_score):
        self.min_frequency = min_frequency
        self.subword_max_length = subword_max_length + 1
        self.minimum_droprate_score = minimum_droprate_score
        self.words = {}
        self.train(corpus_fnames, model_fname)

    def train(self, corpus_fnames, model_fname):
        for n_corpus, fname in enumerate(corpus_fnames):
            corpus_index = fname.split('/')[-1].split('.')[0]
            corpus = DoublespaceLineCorpus(fname, iter_sent=False)
            
            L = self._subword_counting(corpus)
            droprate_scores = self._word_scoring(L)
            for word, score in droprate_scores.items():
                if score < self.minimum_droprate_score:
                    continue
                self.words[word] = max(self.words.get(word, 0), score)
            args = (n_corpus+1, len(corpus_fnames), len(self.words), '%.3f'%get_process_memory())
            print('\r  - updating vocabularies {} / {} corpus, n_vocab={}, mem = {} Gb'.format(*args))
        self._save(model_fname)

    def _subword_counting(self, corpus):
        L = {}
        for n_doc, doc in enumerate(corpus):
            for word in doc.split():
                for e in range(1, min(self.subword_max_length, len(word))+1):
                    subword = word[:e]
                    L[subword] = L.get(subword, 0) + 1
            if n_doc % 1000 == 999:
                print('\r  - scanning ... {} docs'.format(n_doc+1), flush=True, end='')
        return L

    def _word_scoring(self, L):
        droprate_scores = {} 
        for l, count in sorted(L.items(), key=lambda x:len(x[0])):
            if len(l) < 3 or count < self.min_frequency:
                continue
            l_sub = l[:-1]
            droprate = count / L[l_sub]
            droprate_scores[l_sub] = max(droprate_scores.get(l_sub, 0), droprate)
        droprate_scores = {word:1-score for word, score in droprate_scores.items() if len(word) > 1}
        return droprate_scores
    
    def _save(self, model_fname):
        import os
        folder = '/'.join(model_fname.split('/')[:-1])
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(model_fname, 'w', encoding='utf-8') as f:
            for word, score in sorted(self.words.items()):
                f.write('{}\t{}\n'.format(word, score))

class BranchingEntropyDictionaryBuilder:
    def __init__(self, corpus_fnames, model_fname, min_frequency, subword_max_length, minimum_branching_entropy):
        self.min_frequency = min_frequency
        self.subword_max_length = subword_max_length + 1
        self.minimum_branching_entropy = minimum_branching_entropy
        self.words = {}
        self.train(corpus_fnames, model_fname)

    def train(self, corpus_fnames, model_fname):
        for n_corpus, fname in enumerate(corpus_fnames):
            corpus_index = fname.split('/')[-1].split('.')[0]
            corpus = DoublespaceLineCorpus(fname, iter_sent=False)
            
            L = self._subword_counting(corpus)
            entropy = self._word_scoring(L)            
            for word, score in entropy.items():
                if score < self.minimum_branching_entropy:
                    continue
                self.words[word] = max(self.words.get(word, 0), score)
            args = (n_corpus+1, len(corpus_fnames), len(self.words), '%.3f'%get_process_memory())
            print('\r  - updating vocabularies {} / {} corpus, n_vocab={}, mem = {} Gb'.format(*args))
        self._save(model_fname)

    def _subword_counting(self, corpus):
        L = {}
        for n_doc, doc in enumerate(corpus):
            for word in doc.split():
                for e in range(1, min(self.subword_max_length, len(word))+1):
                    subword = word[:e]
                    L[subword] = L.get(subword, 0) + 1
            if n_doc % 1000 == 999:
                print('\r  - scanning ... {} docs'.format(n_doc+1), flush=True, end='')
        return L

    def _word_scoring(self, L):
        from math import log
        from collections import defaultdict
        by_length = defaultdict(lambda: [])
        for l in L:
            n = len(l)
            if len(l) < 3: continue
            by_length[n].append(l)
            
        entropy = {}
        for length, words in by_length.items():
            root_to_branch = defaultdict(lambda: [])
            for word in words:
                root = word[:-1]
                if L[root] < self.min_frequency:
                    continue
                root_to_branch[root].append(L[word])
            for root, counts in root_to_branch.items():
                sum_ = sum(counts)
                probs = [c / sum_ for c in counts]
                probs = [log(p) for p in probs]
                entropy[root] = -sum(probs)
        return entropy
    
    def _save(self, model_fname):
        import os
        folder = '/'.join(model_fname.split('/')[:-1])
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(model_fname, 'w', encoding='utf-8') as f:
            for word, score in sorted(self.words.items()):
                f.write('{}\t{}\n'.format(word, score))

class WordPieceModelBuilder:
    def __init__(self, corpus_fnames, subword_max_length, model_fname, num_units=1000):
        self.num_units = num_units if num_units > 0 else 1000
        self.subword_max_length = subword_max_length
        self.units = {}
        self.max_length = 0
        class Sentence:
            def __init__(self, corpus_fnames):
                self.corpus_fnames = corpus_fnames
            def __iter__(self):
                for i, fname in enumerate(self.corpus_fnames):                    
                    with open(fname, encoding='utf-8') as f:
                        for j, doc in enumerate(f):
                            if j % 100 == 99:
                                print('\r  iterating {} / {} corpus, {} docs ...'.format(i+1, len(self.corpus_fnames), j+1), flush=True, end='')
                            sents = doc.split('  ')
                            if not sents:continue
                            for sent in sents:
                                yield sent.strip()
                    print('\r  iterated {} / {} corpus, mem = {} Gb'.format(i+1, len(self.corpus_fnames), '%.3f'%get_process_memory()), flush=True)
        self._train(Sentence(corpus_fnames))
        self._save(model_fname)
        
    def _train(self, sents):
        def to_subwords(s):
            s = s[:self.subword_max_length]
            s = s.replace('_', '') + '_'
            n = len(s)
            return (s[b:b+r] for b in range(n) for r in range(1, n+1) if b+r <= n)

        def counting(sents):
            from collections import Counter
            return Counter((subword for sent in sents for eojeol in sent.split() for subword in to_subwords(eojeol) if eojeol))

        counter = counting(sents)
        a_syllables = {subword:freq for subword, freq in counter.items() if len(subword) == 1}
        self.units = dict(
            sorted(
                filter(lambda x:len(x[0]) > 1, counter.items()), 
                key=lambda x:(-x[1], -len(x[0]), x[0]))
            [:max(0, self.num_units - len(a_syllables))]
        )
        self.units.update(a_syllables)
        self.max_length = max((len(w) for w in self.units))
        
    def _save(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('num_units={}\n'.format(self.num_units))
            f.write('max_length={}\n'.format(self.max_length))
            for unit, frequency in sorted(self.units.items(), key=lambda x:(-x[1], -len(x[0]))):
                f.write('{}\t{}\n'.format(unit, frequency))
                
class SubwordMatchTokenizer:
    def __init__(self, dictionary_file):
        self.dictionary = self._load(dictionary_file)
        self._max_len = max((len(word) for word in self.dictionary))
        
    def get_words(self):
        return sorted(self.dictionary)
    
    def _load(self, fname):
        with open(fname, encoding='utf-8') as f:
            words = dict((word.split('\t') for word in f))
            words = {word:float(score) for word, score in words.items()}
            return words
        
    def tokenize(self, sent):
        def _tokenize(token):
            words = [token[:e] for e in range(2, min(len(token), self._max_len)+1)]
            words = [word for word in words if word in self.dictionary]
            return words
        
        words = []
        for token in sent.split():
            words += _tokenize(token)
        return words

class WordPieceModelTokenizer:
    def __init__(self, dictionary_file, subword_max_length):
        self._load(dictionary_file)
        self.subword_max_length = subword_max_length
    
    def get_words(self):
        return sorted(self.units)
        
    def _load(self, fname):
        with open(fname, encoding='utf-8') as f:
            try:
                self.num_units = int(next(f).strip().split('=')[1])
                self.max_length = int(next(f).strip().split('=')[1])
            except Exception as e:
                print(e)
            
            self.units = {}
            for row in f:
                try:
                    unit, frequency = row.strip().split('\t')
                    self.units[unit] = int(frequency)
                except Exception as e:
                    print('BPE load exception: {}'.format(str(e)))
                    break
                    
    def tokenize(self, sent):
        words = []
        for token in sent.split():
            words += self._tokenize(token)
        return words
    
    def _tokenize(self, w):
        def initialize(w):
            w = w[:self.subword_max_length]
            w += '_'
            subwords = []
            n = len(w)
            for b in range(n):
                for e in range(b+1, min(n, b+self.max_length)+1):
                    subword = w[b:e]
                    if not subword in self.units:
                        continue
                    subwords.append((subword, b, e, e-b))
            return subwords
        
        def longest_match(subwords):
            matched = []
            subwords = sorted(subwords, key=lambda x:(-x[3], x[1]))
            while subwords:
                s, b, e, l = subwords.pop(0) # str, begin, end, length
                matched.append((s, b, e, l))
                removals = []
                for i, (_, b_, e_, _) in enumerate(subwords):
                    if (b_ < e and b < e_) or (b_ < e and e_ > b):
                        removals.append(i)
                for i in reversed(removals):
                    del subwords[i]
            return sorted(matched, key=lambda x:x[1])
        
        subwords = initialize(w)
        subwords = longest_match(subwords)
        subwords = [s for s, _, _, _ in subwords]
        return subwords