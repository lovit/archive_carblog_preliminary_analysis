from glob import glob
import argparse
import json

def parse(json_object):
    def get_entity(json_object, key):
        return '  '.join(str(json_object.get(key,'')).replace('\t','\n').replace('\r','').strip().split('\n'))
    
    title = get_entity(json_object, 'title')
    content = get_entity(json_object, 'content')
    
    categoryName = get_entity(json_object, 'categoryName')
    sympathyCount = get_entity(json_object, 'sympathyCount')
    writtenTime = get_entity(json_object, 'writtenTime')
    tags = get_entity(json_object, 'tags')
    url = get_entity(json_object, 'url')

    return '{}\t{}'.format(title, content), '{}\t{}\t{}\t{}\t{}'.format(categoryName, tags, sympathyCount, writtenTime, url)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_directory', type=str, default='./', help='json file directory')
    parser.add_argument('--corpus_directory', type=str, default='./', help='corpus directory')
    
    args = parser.parse_args()
    corpus_directory = args.corpus_directory
    car_directories = glob('{}/*'.format(args.json_directory)) # car, year, json
    
    import os
    if not os.path.exists(corpus_directory):
        os.makedirs(corpus_directory)
        print('created directory {}'.format(corpus_directory))
    
    for i_corpus, car_root in enumerate(sorted(car_directories)):
        corpus_name = car_root.split('/')[-1]
        corpus_fname = '{}/{}.txt'.format(corpus_directory, corpus_name)
        index_fname = '{}/{}.index'.format(corpus_directory, corpus_name)
        with open(corpus_fname, 'w', encoding='utf-8') as fc:
            with open(index_fname, 'w', encoding='utf-8') as fi:
                json_fnames = glob('{}/*/*.json'.format(car_root))
                for i, json_fname in enumerate(json_fnames):
                    with open(json_fname, encoding='utf-8') as f:
                        json_object = json.load(f)
                    if not json_object or type(json_object) != dict:
                        continue
                    corpus_content, index_content = parse(json_object)
                    fc.write('{}\n'.format(corpus_content))
                    fi.write('{}\n'.format(index_content))
                    if i % 200 == 199:
                        print('\rparsing ... directory = {}/{}, json = {}/{}{}'.format(i_corpus+1, len(car_directories), i+1, len(json_fnames), ' '*20), flush=True, end='')
        print('\rparsing done. directory = {}/{}, {} complated {}'.format(i_corpus+1, len(car_directories), i+1, ' '*20))

if __name__ == "__main__":
    main()