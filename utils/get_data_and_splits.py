
from multiprocessing.sharedctypes import Value
import os
import sys
sys.path.append('..')
sys.path.append('.')

import pandas as pd
import numpy as np
import re
import _pickle as cPickle

from preprocess.feature_extraction import FeatureExtractor
from preprocess.config import proc_paths
from preprocess.helper import read_pickle_file

from tqdm import tqdm
import joblib

class LoadData:

    def get_data(self, params, extra_data_path=None, extra_data=False):

        if extra_data and not extra_data_path:
            raise ValueError('No extra_data path specified')

        print (params['output_size'],type(params['output_size']))
        #train
        d = np.array(read_pickle_file(proc_paths[params['dataset']]['train']))
        mask = np.array([(d[i][params['emb_type']] != np.array([0]*params['output_size']).astype('float32')).all() for i in tqdm(range(len(d)))])
        train = pd.DataFrame.from_records(d[mask])

        #val
        d = np.array(read_pickle_file(proc_paths[params['dataset']]['dev']))
        mask = np.array([(d[i][params['emb_type']] != np.array([0]*params['output_size']).astype('float32')).all() for i in tqdm(range(len(d)))])
        val = pd.DataFrame.from_records(d[mask])

        #test
        d = np.array(read_pickle_file(proc_paths[params['dataset']]['test']))
        mask = np.array([(d[i][params['emb_type']] != np.array([0]*params['output_size']).astype('float32')).all() for i in tqdm(range(len(d)))])
        test = pd.DataFrame.from_records(d[mask])
        
                
        if extra_data:
            #get extra data
            with open(extra_data_path, "rb") as f:
                extra_data = np.array(cPickle.load(f))

            zero_array = np.array([(extra_data[i][params['emb_type']] != np.array([0]*params['output_size']).astype('float32')).all() for i in tqdm(range(len(extra_data)))])

            extra_data = pd.DataFrame.from_records(extra_data)

            extra_data = extra_data[zero_array]
        
            train = pd.concat([train, extra_data], axis = 0).reset_index(drop=True)
        
        
        
        #remove words in val, test from train
        train = train[~(train.word.isin(set(val.word)))]
        train = train[~(train.word.isin(set(test.word)))].reset_index(drop=True)
        
        


#         train['split_gloss'] = train.gloss.apply(lambda x: x.split(';'))
#         val['split_gloss'] = val.gloss.apply(lambda x: x.split(';'))
#         test['split_gloss'] = test.gloss.apply(lambda x: x.split(';'))

#         train = train.explode('split_gloss')
#         val = val.explode('split_gloss')
#         test = test.explode('split_gloss')
        
        
        
#         train = train[['word','split_gloss','fasttext']].reset_index(drop=True)
#         train.columns = ['word','gloss','fasttext']

#         val = val[['word','split_gloss','fasttext']].reset_index(drop=True)
#         val.columns = ['word','gloss','fasttext']

#         test = test[['word','split_gloss','fasttext']].reset_index(drop=True)
#         test.columns = ['word','gloss','fasttext']
        
    
        train = train[['word', 'gloss', params['emb_type']]]
        val = val[['word', 'gloss', params['emb_type']]]
        test = test[['word', 'gloss', params['emb_type']]]
        
        print(f'Len of Train before drop duplicates - {len(train)}')
        print(f'Len of Val before drop duplicates - {len(val)}')
        print(f'Len of Test before drop duplicates - {len(test)}')
        
        train = train.drop_duplicates(['gloss', 'word']).reset_index(drop=True)
        val = val.drop_duplicates(['gloss', 'word']).reset_index(drop=True)
        test = test.drop_duplicates(['gloss', 'word']).reset_index(drop=True)
        
        
        
        
        print(f'Len of Train - {len(train)}')
        print(f'Len of Val - {len(val)}')
        print(f'Len of Test - {len(test)}')
        
        # joblib.dump(train, 'data/clean_data/train.joblib')
        joblib.dump(train, '../data/clean_data/train.joblib')
        joblib.dump(val, '../data/clean_data/val.joblib')
        joblib.dump(test, '../data/clean_data/test.joblib')
        
        
        return train, val, test
        
        
        