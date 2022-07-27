import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import sys
sys.path.append('..')
from preprocess.feature_extraction import FeatureExtractor
from preprocess.config import proc_paths
from preprocess.helper import read_pickle_file
from tqdm import tqdm



print ('Reading train, val, test....')
train = read_pickle_file(proc_paths['train'])
dev = read_pickle_file(proc_paths['dev'])
test = read_pickle_file(proc_paths['test'])

print (f'train len: {len(train)} dev len: {len(dev)}')
print ('\n')

# ex_lengths = np.array([len(j.split()) for i in train for j in i['example']])
# gloss_lengths = np.array([len(i['gloss'].split()) for i in train])
# sns.distplot(gloss_lengths)

def remove_words_from_train(train, no_train):
    '''
    Remove words from train which are present in dev and test
    '''
    
    print ('Train corpus before removal: ',len(train))

    no_train_words = set([i['word'] for i in no_train])
    print ('non_train words in train: ',len(no_train_words))

    #start removing words
    _train = [i for i in train if i['word'] not in no_train_words]
    print ('Train corpus after removal: ',len(_train))

    return _train

train = remove_words_from_train(train,dev) #Remove words from train which are present in dev and test
print ('\n')
train = remove_words_from_train(train,test) #Remove words from train which are present in dev and test
print ('\n')


def preprocess(sent, do_lower=False, max_len=None, add_spec_toks=False):
    '''
    preprocess sentence to lowercase
    shorten to max_len 
    add special tokens like BOS, EOS
    '''
    
    if do_lower:
        sent = sent.lower()
    if isinstance(sent, str):
        sent = sent.split()
    if max_len:
        sent = sent[:max_len]
    if add_spec_toks:
        sent = ['<BOS>']+sent+['<EOS>']
    return (' ').join(sent)
    

GLOSS_MAXLEN = 33
EXAMPLE_MAXLEN = 80 

def get_wge_triple(data):
    '''
    Get word gloss example triple
    ### CHECK LOWERCASING
    wg pair can have multiple examples - each is considered different datapoint
    => 1 WG 3 Ex --> 3 Datapoints
    '''
    
    glosses, examples, words = [],[],[]
    for item in tqdm(data, total = len(data)):
        gloss = preprocess(item['gloss'], max_len = GLOSS_MAXLEN)
        word = item['word']

        for ex in item['example']:
            ex  = preprocess(ex, max_len=EXAMPLE_MAXLEN)
            
            glosses.append(gloss)
            words.append(word)
            examples.append(ex)


    return words, examples, glosses

print ('Extracting word, gloss, example triple')
train = get_wge_triple(train)
dev = get_wge_triple(dev)

print (f'Len train: {len(train[0])} dev: {len(dev[0])}')

ridx = random.choice(range(0,len(train[0])))
print(f'Random Index: {ridx}, \nWord: {train[0][ridx]}')
print (f'Example: {train[1][ridx]},\nGloss: {train[2][ridx]}')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<BOS>', eos_token='<EOS>') #gpt2-medium
print (len(tokenizer))

special_tokens_dict = {'additional_special_tokens': ['<DEF>','<EPL>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print (len(tokenizer))

# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

tokenizer.padding_side = "left"

# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

model.resize_token_embeddings(len(tokenizer))

print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))
print ()

batch_size = 4
class GPT2Dataset(Dataset):

  def __init__(self, sentences, tokenizer, max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for sent in sentences:
      encodings_dict = tokenizer(sent.split(), truncation=True, max_length=max_length,
              padding="max_length", is_split_into_words=True)

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 



def build_prompt(data):
    sents = []
    for word, gloss, example in zip(data[0],data[1],data[2]):
        s = f'<BOS> The meaning of {word} in <EPL> {example} is <DEF> {gloss} <EOS>'
        sents.append(s)
    return sents

train_prompts = build_prompt(train)
dev_prompts = build_prompt(dev)
print (random.choice(train_prompts))
print (random.choice(dev_prompts))

print ('\nBuilding tokenized dataset .....')
train_dataset = GPT2Dataset(train_prompts, tokenizer=tokenizer, max_length=768)
dev_dataset = GPT2Dataset(dev_prompts, tokenizer=tokenizer, max_length=768)

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            dev_dataset, # The validation samples.
            sampler = SequentialSampler(dev_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# some parameters I cooked up that work reasonably well

epochs = 1
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 100

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


total_t0 = time.time()

training_stats = []

model = model.to(device)

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            model.eval()

            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()

        loss.backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


output_dir = f'checkpoints/gpt2_exp2/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)