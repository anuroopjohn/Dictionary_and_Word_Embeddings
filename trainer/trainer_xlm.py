
import os
import sys
#os.environ['HF_HOME'] = '/data/users/ugarg/hf/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = '/data/users/ugarg/hf/hf_cache/'
os.environ['CUDA_VISIBLE_DEVICES']='3'

#sys.path.append('../../..')
sys.path.append('../')
sys.path.append('./')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)
from torch.nn.functional import one_hot


from collections import Counter

import pandas as pd
import torch
import numpy as np
import random
from transformers import AdamW, AutoTokenizer,  AutoModel
from torch.nn.functional import one_hot
from collections import Counter
from torch.optim import AdamW, Adam
from transformers import get_scheduler, get_cosine_with_hard_restarts_schedule_with_warmup

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)
import joblib
import datetime
from tqdm.auto import tqdm


from utils.get_data_and_splits import LoadData
from utils.params import get_roberta_params, get_xlm_params
from utils.model import Model
from utils.utils import checkpoint_builder

from CreatePytorchDataset import TrainDataset, ValDataset

loaddata = LoadData()
train, val, test = loaddata.get_data(
        extra_data=False,
        extra_data_path="/data/users/abose1/Capstone/Dictionary_and_Word_Embeddings/data/static_wiki.pkl"       
        ) #removes non-zero,

print (train[['word','gloss', 'fasttext']].info())

print (train.sample(6))

params = get_xlm_params()

print (f'Params: {params}')


##################
#   tokenizer
##################

from transformers import AutoTokenizer

model_checkpoint = params['model_checkpoint']
print(f'\nLoading Tokenizer: {model_checkpoint} ...')
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# save_checkpoint_path = f"../checkpoints/{params['model_checkpoint']}_{params['loss_fn_name']}loss_{params['emb_type']}_embs_{params['use_adapters']}_adapter_{params['extra_data']}_extradata"
save_checkpoint_path = f"../checkpoints/{params['model_checkpoint']}"

def get_path(save_checkpoint_path,i=0):
    if os.path.exists(save_checkpoint_path+'_'+str(i)):
        return get_path(save_checkpoint_path,i+1)
        # print (f'Path {save_checkpoint_path} already exists.')
    else:
        return save_checkpoint_path+'_'+str(i)

save_checkpoint_path = get_path(save_checkpoint_path,0)   
os.mkdir(save_checkpoint_path)

print(f'Save checkpoint path: {save_checkpoint_path}')

import json
json.dump(params,open(save_checkpoint_path+'/params.json','w'),indent=4)


train_dataset = TrainDataset(
                    train, 
                    tokenizer, 
                    params['source_column'],
                    params['max_len'], 
                    params['emb_type']
                )

val_dataset = ValDataset(
                    val, 
                    tokenizer, 
                    params['source_column'],
                    params['max_len'], 
                    params['emb_type']
                )

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn = data_collator,
    batch_size = params['batch_size'],
)
val_dataloader = DataLoader(
    val_dataset, 
    collate_fn = data_collator, 
    batch_size=params['batch_size']
)
print (f'Train len {len(train_dataloader)}, val len: {len(val_dataloader)}')


####################################
#   Load Model
####################################

print(f"\nUsing device {params['device']} for training...")

model = Model(params['model_checkpoint'], 
              params['output_size'], 
              params['dropout'], 
              params['device'], 
              params['loss_fn_name'], 
              params['use_adapters'])
model.to(params['device'])
# print (model)


####################################
#   Create Optimizer
####################################
print(f"\nCreating optimizer with Learning Rate: {params['learning_rate']}")


optimizer = AdamW(model.parameters(), lr = params['learning_rate'])

print(f"\nTraining for {params['num_epochs']} epochs with early stopping set to {params['early_stopping_limit']}\n\n")

num_train_epochs = params['num_epochs']
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=params['num_warmup_steps'],
    num_training_steps=num_training_steps,
)

if params['resume_from_checkpoint']:
    print('Resuming from Checkpoint...')
    print('Loading Checkpoint ...')
    checkpoint = torch.load(save_checkpoint_path+'/model.pt', map_location=params['device'])
    print('Setting Models state to checkpoint...')
    #assert checkpoint['model_params'] == model_params
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model state set.')
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    print('Optimizer state set.')

print (f'num_training steps : {num_training_steps}')
update_steps=1000
progress_bar = tqdm(range(num_training_steps//update_steps))
early_stopping_counter = 0
early_stopping_limit = params['early_stopping_limit']
best_valid_loss = float('inf')

update=0
for epoch in range(num_train_epochs):
    train_loss=0
    valid_loss =0
    

    
    print(f"[Epoch {epoch} / {num_train_epochs}]")
    # Training
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        loss, out, actual = model(batch)
        #loss = outputs.loss
        loss.backward(loss)#, retain_graph = True)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))

        update+=1
        if update % update_steps == 0:
            progress_bar.update(1)
            progress_bar.set_postfix(loss = train_loss)
        
        
    
    
    # Evaluation
    model.eval()
    for batch_idx, batch in enumerate(val_dataloader):
        with torch.no_grad():
            loss, out, actual = model(batch)


        labels = actual
        predictions = out
        
        valid_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))
        

    print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))

    #early stopping, checkpointing
    if valid_loss < best_valid_loss:
        early_stopping_counter = 0
        best_valid_loss = valid_loss
        #create checkpoint
        print("Loss improved saving checkpoint... ")
        checkpoint_builder(model, optimizer, epoch, save_checkpoint_path+'/model.pt')

    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_limit:
            print(f'\nLoss did not reduce for last {early_stopping_counter} epochs. Stopped training..')
            break

