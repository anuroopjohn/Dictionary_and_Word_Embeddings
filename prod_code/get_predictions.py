import sys
sys.path.append('../')
import numpy as np
from trainer.params import get_roberta_params, get_xlm_params
from utils.predict import prep_model, predict_on_batch
from utils.faiss_utils import create_and_store_index, get_top_n_accuracy
from tqdm.auto import tqdm
import json
import faiss

class GetPredictions:
    
    def __init__(self, save_checkpoint_path, model_type = 'monolingual'):
        
        # if model_type == 'multilingual':
        #     self.params = get_xlm_params()
        # elif model_type == 'monolingual':
        #     self.params = get_roberta_params()
        self.save_checkpoint_path = save_checkpoint_path

        print('Preparing Model and Search Attributes...')
        self.params = self.load_saved_params()
        self.prepare_model()

    def load_saved_params(self):
        return json.load(open(f'{self.save_checkpoint_path}/params.json','r'))

    def prepare_model(self):
        
        self.tokenizer, self.model, self.data_collator, self.sim,\
                self.faiss_index, self.faiss_idx_word_lookup = prep_model(self.params, f'{self.save_checkpoint_path}/model.pt')

        self.model.eval()
        
    def _get_rank(self, actual_word, preds, n_nearest_num):
        '''
        Find index word in list of words
        if not found, return n_nearest_num + 1
        '''
        
        try:
            rank = preds.index(actual_word)
        except ValueError:
            rank = n_nearest_num
        return rank + 1 #rank starts from 1 not zero

    def get_predictions(self, data, NEAREST_NEIGHBOUR_NUM = 1000, pred_batch = 1):

        data = data.reset_index(drop=True)
        #Use it if text in list
        #NEAREST_NEIGHBOUR_NUM = 1000
        
        
        #pred_batch = 50
        ranks = []
        all_preds = []
        for i in tqdm(range(0, len(data), pred_batch)):
            x = data.loc[i:i+pred_batch-1] #-1 because iloc give last index also as element

            preds = predict_on_batch(
                x['gloss'].values.tolist(), 
                self.tokenizer, self.model, self.data_collator, self.sim, self.faiss_index, 
                self.faiss_idx_word_lookup, NEAREST_NEIGHBOUR_NUM
            ) # nearest neighbour words for batch of text (gloss)
            
            actual_word = x['word'].values #gold word 
            
            for j in range(len(preds)): # O(batchsize)
                pred = [self.faiss_idx_word_lookup[k] for k in preds[j]]
                
                # pred = list(dict.fromkeys(pred))  #remove repeat words keeping sort order
                
                all_preds.append(pred[:100]) #return top-k nearest words for a gloss

                rank = self._get_rank(actual_word[j], pred, NEAREST_NEIGHBOUR_NUM)
                ranks.append(rank)
            
                

        print(f"Average Rank @{NEAREST_NEIGHBOUR_NUM} : {np.mean(ranks)}")
        print(f"Median Rank @{NEAREST_NEIGHBOUR_NUM} : {np.median(ranks)}")
        print(f"Rank Variance @{NEAREST_NEIGHBOUR_NUM} : {np.std(ranks)}")
        
        for n in [1, 2 ,5,10,50,100]:
            print(f"Top {n} accuracy is {get_top_n_accuracy(ranks, n)}")
            
        return ranks, all_preds
        
        
    def get_prediction_on_sentence(self, text, NEAREST_NEIGHBOUR_NUM = 50):
        preds = predict_on_batch(
            [text], 
            self.tokenizer, self.model, self.data_collator, self.sim, self.faiss_index, 
            self.faiss_idx_word_lookup, NEAREST_NEIGHBOUR_NUM
        )
        preds = [self.faiss_idx_word_lookup[i] for i in preds[0]]
        return list(dict.fromkeys(preds))
    
    
    