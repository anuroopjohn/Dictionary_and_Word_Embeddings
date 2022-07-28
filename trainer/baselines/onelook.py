import sys
sys.path.append('../..')

import joblib
from tqdm import tqdm
import requests
import numpy as np
from utils.faiss_utils import get_top_n_accuracy
import os.path


def onelook_gloss_to_word_lookup(data_path, save_path):
    '''
    Use datamuse api (onelook url) to obtain words given a gloss/concept

    params:
        data_path: path to dataset with list of {word,gloss} pairs. Only glosses are used for prediction
        save_path: path for saving predictions (top-100 closest words)

    returns:
        onelook_preds: json object with word, gloss, pred_word (100 words)
    '''

    if os.path.isfile(save_path):
        print (f'Predictions already available in {save_path}...')
        return joblib.load(save_path)

    data = joblib.load(data_path)

    onelook_preds = []
    not_found=[]

    for gloss,word in tqdm(zip(data['gloss'].tolist(),data['word'].tolist()),total=len(data)):

        request = requests.get(f"https://api.datamuse.com/words?ml={gloss}")

        # decode json
        response = request.json()
        
        cur_preds = []
        try:
            for i in response: cur_preds.append(i['word'])
        except:
            not_found.append(word)
            print (f'{word} with gloss: "{gloss}" not found')

        onelook_preds.append({'word':word,'gloss':gloss,'pred_word':cur_preds})
    
    print (f'{len(not_found)} glosses have no corresponding word out of {len(data)}')

    joblib.dump(onelook_preds, save_path)

    return onelook_preds

def index_of(val, in_list, N_NEIGHBORS=100):
    try:
        return in_list.index(val)+1
    except ValueError:
        return N_NEIGHBORS+1

def score_predictions(preds):
    '''
    Calculates average, median and top-k accuracy given gold word and N nearest wordlist
    '''

    ranks = []
    for i in preds:
        ranks.append(index_of(i['word'].lower(),i['pred_word']))

    ranks = np.array(ranks)
    print(f"Average Rank @{100} : {np.mean(ranks)}")
    print(f"Median Rank @{100} : {np.median(ranks)}")
    print(f"Rank Variance @{100} : {np.std(ranks)}")

    for n in [1, 2 ,5,10,50,100]:
        print(f"Top {n} accuracy is {get_top_n_accuracy(ranks, n)}")
            
if __name__ == '__main__':

    mode = ['val','test'][1]
    pred_path = f'./preds/onelook_{mode}_preds.joblib'

    ### use api to get predictions for words in data
    data_path = f'../../data/clean_data/{mode}.joblib'

    preds = onelook_gloss_to_word_lookup(data_path=data_path, save_path=pred_path)

    score_predictions(preds)