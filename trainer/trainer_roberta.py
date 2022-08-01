
import os
import sys
#os.environ['HF_HOME'] = '/data/users/ugarg/hf/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = '/data/users/ugarg/hf/hf_cache/'
os.environ['CUDA_VISIBLE_DEVICES']='1'

#sys.path.append('../../..')
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)
from torch.nn.functional import one_hot


from collections import Counter

from datasets import load_dataset, Dataset, DatasetDict
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




from trainer.params import get_roberta_params
from utils.model import Model
from utils.utils import checkpoint_builder


from CreatePytorchDataset import TrainDataset, ValDataset






train = joblib.load('../clean_data/train.joblib')
val = joblib.load('../clean_data/val.joblib')
test = joblib.load('../clean_data/test.joblib')




print(f"Len of train = {len(train)}")
print(f"Len of dev = {len(val)}")
print(f"Len of test = {len(test)}")

params = get_roberta_params()



##################
#   tokenizer
##################

from transformers import AutoTokenizer

model_checkpoint = params['model_checkpoint']
print(f'\nLoading Tokenizer: {model_checkpoint} ...')
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


save_checkpoint_path = f"../checkpoints/{params['model_checkpoint']}_model_{params['loss_fn_name']}_loss_{params['emb_type']}_embs_{params['use_adapters']}_adapter.pt"



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
    num_warmup_steps=300,
    num_training_steps=num_training_steps,
)




progress_bar = tqdm(range(num_training_steps))
early_stopping_counter = 0
early_stopping_limit = params['early_stopping_limit']
best_valid_loss = float('inf')

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
        checkpoint_builder(model, optimizer, epoch, save_checkpoint_path)

    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_limit:
            print(f'\nLoss did not reduce for last {early_stopping_counter} epochs. Stopped training..')
            break


            
print('TRAINING DONE...')