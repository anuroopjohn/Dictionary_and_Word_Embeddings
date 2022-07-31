import numpy as np
import io
import sys
sys.path.append('..')
from preprocess.config import proc_paths
from preprocess.helper import read_pickle_file
from preprocess.helper import write_pickle_file

def load_vec(emb_path, nmax=None):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            # if len(word2id) == nmax:
            #     break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def add_muse_to_data(data, embs, word2id):
    '''
    Add a key, value pair to data
    key: muse
    value: [muse_vector]
    '''
    zero_vec = np.zeros(embs[0].shape)
    not_found=[]

    for idx,i in enumerate(data):
        word = i['word'].lower()
        if word in word2id:
            data[idx]['muse'] = [embs[word2id[word]]]
        else:
            data[idx]['muse'] = [zero_vec]
            not_found+=[word]
    print (f'{len(not_found)} not found in {len(data)}')
    return data

if __name__=='__main__':
    train = read_pickle_file(proc_paths['train'])
    dev = read_pickle_file(proc_paths['dev'])
    test = read_pickle_file(proc_paths['test'])

    print ('Loading muse embedding from file....')
    embs, id2word, word2id = load_vec('../data/muse_embs/wiki.multi.en.vec')

    train = add_muse_to_data(train,embs,word2id)
    dev = add_muse_to_data(dev,embs,word2id)
    test = add_muse_to_data(test,embs,word2id)

    write_pickle_file(train,proc_paths['train'])
    write_pickle_file(dev,proc_paths['dev'])
    write_pickle_file(test,proc_paths['test'])


