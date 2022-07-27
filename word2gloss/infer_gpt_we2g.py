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


os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import sys
sys.path.append('..')

from preprocess.feature_extraction import FeatureExtractor
from preprocess.config import proc_paths
from preprocess.helper import read_pickle_file
from tqdm import tqdm



# output_dir = f'checkpoints/gpt2_exp2/'
output_dir = f'checkpoints/gpt2-medium_exp1/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import logging
logging.basicConfig(filename=f'{output_dir}/log_inference.txt',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


logging.info('Reading train, val, test....')
# train = read_pickle_file(proc_paths['train'])
dev = read_pickle_file(proc_paths['dev'])
# test = read_pickle_file(proc_paths['test'])

logging.info(f'dev len: {len(dev)}')
logging.info('\n')

# ex_lengths = np.array([len(j.split()) for i in train for j in i['example']])
# gloss_lengths = np.array([len(i['gloss'].split()) for i in train])
# sns.distplot(gloss_lengths)


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
LOWERCASE = False

logging.info(f'LOWERCASE :{LOWERCASE}')

def get_wge_triple(data):
    '''
    Get word gloss example triple
    ### CHECK LOWERCASING
    wg pair can have multiple examples - each is considered different datapoint
    => 1 WG 3 Ex --> 3 Datapoints
    '''
    
    glosses, examples, words = [],[],[]
    for item in tqdm(data, total = len(data)):
        gloss = preprocess(item['gloss'], max_len = GLOSS_MAXLEN, do_lower=LOWERCASE)
        word = item['word']

        for ex in item['example']:
            ex  = preprocess(ex, max_len=EXAMPLE_MAXLEN, do_lower=LOWERCASE)
            
            glosses.append(gloss)
            words.append(word)
            examples.append(ex)


    return words, examples, glosses

logging.info('Extracting word, gloss, example triple')
# train = get_wge_triple(train)
dev = get_wge_triple(dev)

logging.info(f'len dev: {len(dev[0])}')

model_checkpoint = ['gpt2','gpt2-medium'][1]
logging.info(f'MODEL NAME: {model_checkpoint}')

tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, bos_token='<BOS>', eos_token='<EOS>', pad_token = '<PAD>') #gpt2-medium
logging.info(f'Tokenizer length: {len(tokenizer)}')

special_tokens_dict = {'additional_special_tokens': ['<DEF>','<EPL>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
logging.info(f'Tokenizer length after spec toks: {len(tokenizer)}')

# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained(output_dir, output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained(output_dir, config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

# tokenizer.padding_side = "left"

# Define PAD Token = EOS Token = 50256
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = model.config.eos_token_id

model.resize_token_embeddings(len(tokenizer))

logging.info("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
logging.info("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
logging.info("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
logging.info("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))
logging.info('\n')


def build_prompt(data):
    '''
    desc: 
        convert word + example into template
    input:
        (list of word, list of example, list of gloss)
    output:
        sentences as per the templates
    '''

    sents = []
    for word, example, gloss in zip(data[0],data[1],data[2]):
        s = f'<BOS> The meaning of {word} in <EPL> {example} is <DEF> ' #{gloss} <EOS>'
        sents.append(s)
    return sents

dev_prompts = build_prompt(dev)
logging.info(random.choice(dev_prompts))
print ('dev_prompts len',len(dev_prompts), len(dev[0]))

top_k = 0
max_length = 200
top_p=0.7
num_return_sequences=3

batch_size = 4
logging.info(f'batch size: {batch_size}')
logging.info(f'top_k: {top_k}')
logging.info(f'top_p: {top_p}')
logging.info(f'max_length: {max_length}')
logging.info(f'num_return_sequences: {num_return_sequences}')

def generation(dev_prompts, top_k, max_length, top_p, num_return_sequences):

    batch_out_sentence = []
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    for i in tqdm(range(0,len(dev_prompts),batch_size)):
        sentences = dev_prompts[i:i+batch_size]
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)

        torch.manual_seed(0)
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            do_sample=True,
            top_k=top_k,
            max_length = max_length,
            top_p=top_p,
            num_return_sequences=num_return_sequences)

        batch_out_sentence += tokenizer.batch_decode(outputs, skip_special_tokens=False)
        # print (len(batch_out_sentence),len(sentences),i)
        # break
    return batch_out_sentence

batch_out_sentence = generation(dev_prompts=dev_prompts, top_k=top_k, top_p=top_p, 
        max_length=max_length, num_return_sequences=num_return_sequences)
print (f'len(batch_out_sentence) {len(batch_out_sentence)}')

preds = []
for i,sent in enumerate(batch_out_sentence):
    
    exmpl = dev[1][i//num_return_sequences]
    sent = sent[sent.find('<DEF>')+6:]
    sent = sent[:sent.find('<EOS>')]
    sent = (' ').join(sent.split())
    preds.append(sent)

preds = [preds[i:i+num_return_sequences] for i in range(0,len(preds),num_return_sequences)]
print (f'len(preds): {len(preds)}')

assert len(preds) == len(dev[0])

# Create output directory if needed
if not os.path.exists(f'{output_dir}/preds'):
    os.makedirs(f'{output_dir}/preds')

df = pd.DataFrame(zip(dev[0],dev[1],dev[2],preds),columns = ['word','example','gold_gloss','pred_gloss'])
df.to_json(f'{output_dir}/preds/dev_{max_length}_{top_p}_{top_k}.json', orient='records')
print (df.head(2))
