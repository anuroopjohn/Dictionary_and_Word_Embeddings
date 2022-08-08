import os
import sys
#os.environ['HF_HOME'] = '/data/users/ugarg/hf/hf_cache/'
# os.environ['TRANSFORMERS_CACHE'] = '/data/users/ugarg/hf/hf_cache/'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

sys.path.append('../../../')
import pandas as pd
import numpy as np

import re
from tqdm import tqdm



import torch

import random
from transformers import AdamW, AutoTokenizer,  AutoModel
from torch.nn.functional import one_hot
from collections import Counter


import joblib
import datetime
from tqdm import tqdm


import faiss

from trainer.params import get_roberta_params
from evaluation.similarity import Similarity
from utils.faiss_utils import load_data, create_and_store_index, get_top_n_accuracy
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding




from utils.model import Model

from transformers import AutoTokenizer


def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

# project_path = "/data/users/ugarg/capstone/code/Dictionary_and_Word_Embeddings-main/gloss2word/v2/"

def prep_model(params, save_checkpoint_path=None, index_name = None):



    ##################
    #   tokenizer
    ##################

    
    
    print(f"\nLoading Tokenizer: {params['model_checkpoint']} ...")
    tokenizer = AutoTokenizer.from_pretrained(params['model_checkpoint'])
    # save_checkpoint_path = project_path + f"checkpoints/xlm-roberta-large_model_cosine_loss_fasttext_embs_True_adapter0.0001_lr.pt"#project_path + f"checkpoints/{params['model_checkpoint']}_model_{params['loss_fn_name']}_loss_{params['emb_type']}_embs_{params['use_adapters']}_adapter.pt"

    print(f"Loading from : {save_checkpoint_path}")

    
    print(f"\nUsing device {params['device']} for training...")

    
    model = Model(params['model_checkpoint'], 
              params['output_size'], 
              params['dropout'], 
              params['device'], 
              params['loss_fn_name'], 
              params['use_adapters'])
    model.to(params['device'])
    model.eval()
    

    print('Loading Checkpoint ...')
    checkpoint = torch.load(save_checkpoint_path, map_location=params['device'])
    print('Setting Models state to checkpoint...')
    
    #assert checkpoint['model_params'] == model_params
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model state set.')
    #optimizer.load_state_dict(checkpoint['optim_state_dict'])
    #print('Optimizer state set.')
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    faiss_index = faiss.read_index(f'../data/clean_data/faiss_index/{index_name}/index_{emb_type}_{knn_measure}')
    faiss_idx_index_to_word_lookup = joblib.load(f'../data/clean_data/faiss_index/{index_name}/word_lookup_{emb_type}_{knn_measure}.joblib')#faiss_idx_index_to_word_lookup.joblib')
    print (f'Index dict len: {len(faiss_idx_index_to_word_lookup)}')
    
    sim = Similarity()
    
    return tokenizer, model, data_collator, sim, faiss_index, faiss_idx_index_to_word_lookup
 


    
    

def predict_on_batch(texts, tokenizer, model, data_collator, sim, faiss_index, faiss_idx_word_lookup, n_nearest_num):
    """
    desc:
        Using the input text we predict the embedding
        Then using the embedding, we get the nearest neighbours
        
    args:
        text: list of str: Text for which embedding is to be predicted
        sim: Similarity object: 
        faiss_index: FAISS index
        faiss_idx_word_lookup: index word lookup to get words corresponding to faiss nearest neighbours
        n_nearest_num: number of nearest neighbour to return
        
    returns:
        nearest neighbour words for text (gloss)
    """
    
    tokenized_data = tokenizer(texts)
    
    tokenized_data = data_collator(tokenized_data)
    
    out = model(tokenized_data, mode = 'Test')
    
    
   
    out = out.detach().cpu().numpy()
    
    _, pred_sim_idx = sim.get_knn(faiss_index, n_nearest_num , out)

    
    # nearest_n = []
    # for i in tqdm(range(len(out))):
    #     preds = [faiss_idx_word_lookup[j] for j in pred_sim_idx[i][:100]]
    #     nearest_n.append( preds)
        
    return pred_sim_idx

    
    
    
    
    

    
    
