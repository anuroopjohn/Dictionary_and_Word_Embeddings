from plistlib import load
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import json
import random
import seaborn as sns
import pickle
import string
import math
tqdm.pandas()
from perplexity import load_gpt_model,gpt_score

root_path = '/data/users/kartik/Dictionary_and_Word_Embeddings/rareword_project/'


# books1_files = glob.glob('./data/books1/epubtxt/*.txt')
# print ('Number of documents in Book Corpus: ',len(books1_files))
# lines = []
# for i,file in tqdm(enumerate(books1_files)):
#     with open(file) as f:
#         lines+= f.readlines()
#     # if i%1000 ==0:
#     #     print ('Num lines: ',len(lines))
    
# print (f'Total number of lines in Book Corpus: {len(lines)}')

# def clean_corpus(lines):
#     _lines  = []
#     for line in tqdm(lines):
#         line = line.rstrip('\n') #remove newline
#         line = line.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
#         line = line.lower() #remove casing
#         line = (' ').join(line.split()) #remove extra spaces
#         _lines.append(line)
#     return _lines
# lines = clean_corpus(lines)
# print (len(lines))

# with open('./data/bookcorpus1.txt','w') as f:
#     for line in lines:
#         f.write(line+'\n')

# with open('./data/bookcorpus1.txt') as f:
#     lines = [line.rstrip('\n') for line in f.readlines()]

# from collections import Counter
# counts = Counter([j for i in lines for j in i.split()])
# freq_sorted = counts.most_common()
# print (len(freq_sorted))

# # rank_dict = {word[0]:i+1 for i,word in enumerate(freq_sorted)}
# with open('./data/bookcorpus1_freqdict.pkl','wb') as f:
#     pickle.dump(freq_sorted,f)

print ('loading frequency dict from book corpus...')
with open(root_path+'data/bookcorpus1_freqdict.pkl','rb') as f:
    freq_sorted = pickle.load(f)

print ('top 5 words: ', freq_sorted[:5])
print ('lowest 10 words: ',freq_sorted[-10:])


def load_glue_inference_file(path):
    
    df = pd.read_csv(path)
    return df

def calculate_accuracy(preds, golds, task):
    
    assert len(golds)==len(preds)

    if task == 'mnli':
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(golds, preds)
    else:
        accuracy = [preds[i]==golds[i] for i in range(len(preds))]
        return sum(accuracy)/len(accuracy)

def calculate_sentence_rarity_log(sentence, freq_dict):

    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    words = sentence.split()

    rarity = []

    for word in words:
        if word not in freq_dict:
            word_freq = 0
            # print (f'word: {word} not in dictionary')
            epsilon = 0.99
        else:
            epsilon = 0
            word_freq = freq_dict[word]
        
        rarity += [1/(math.log(word_freq+1,2)+epsilon)]
    # print (list(zip(words,rarity))) 
    return sum(rarity)/len(words)

def calculate_sentence_rarity_perplexity(sentence, model, tokenizer):

    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    rarity = gpt_score(sentence, model=model, tokenizer=tokenizer)
    return rarity

def save_inference_to_file(df, path):
    
    df.to_csv(path, index=False)

def divide_into_chunks_by_RI(df,intervals):
    chunks = []
    for start,end in zip(intervals[:-1],intervals[1:]):
        temp = df[df.ri.between(start,end)]
        chunks.append(temp)
    return chunks

def pipeline(task, freq_dict):
    
    print ('Running Inference....')

    load_path = f'./glue_preds/{task}_bertft.csv'
    df = load_glue_inference_file(load_path)

    gold_col = {'qnli':'label','sst':'label','cola':'label',
        'wnli':'label','mnli':'gold_label','qqp':'is_duplicate','rte':'label'}

    if task =='qnli':
        itos = ['entailment','not_entailment']
        qns, sents, gold_labels = df['question'].tolist(), df['sentence'].tolist(), df['label'].tolist()
        df['proc_sent'] = df['question'].apply(lambda x: x+' ')+df['sentence']

    elif task == 'sst':
        itos = ['0','1']
        sents, gold_labels = df['sentence'].tolist(), df['label'].tolist()
        df['proc_sent'] = df['sentence']

    elif task == 'cola':
        itos = ['0','1']
        sents, gold_labels = df['sentence'].tolist(), df['label'].tolist()
        df['proc_sent'] = df['sentence']
    
    if task =='wnli':
        itos = ['0','1']
        sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['label'].tolist()
        df['proc_sent'] = df['sentence1'].apply(lambda x: x+' ')+df['sentence2']
   
    if task =='mnli':
        itos = ['neutral','contradiction','entailment']
        stoi = {k:v for v,k in enumerate(itos)}

        sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['gold_label'].tolist()
        df['proc_sent'] = df['sentence1'].apply(lambda x: x+' ')+df['sentence2']
        
        gold_labels = [stoi[i] for i in gold_labels]
        preds = [stoi[i] for i in preds]

    if task =='qqp':
        itos = ['0','1']
        sents1, sents2, gold_labels = df['question1'].tolist(), df['question2'].tolist(), df['is_duplicate'].tolist()
        df['proc_sent'] = df['question1'].apply(lambda x: x+' ')+df['question2']
        
    if task =='rte':
        itos = ['entailment','not_entailment']
        sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['label'].tolist()
        df['proc_sent'] = df['sentence1'].apply(lambda x: x+' ')+df['sentence2']
   
    # if task =='sts':
    #     itos = ['entailment','not_entailment']
    #     sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['label'].tolist()
    #     preds, confidence = run_nli_inference(sents1, sents2, itos, model, tokenizer)

    preds = df['preds'].tolist()
    
    print ('Calculating RI...')
  
    # df['ri'] = df['proc_sent'].progress_apply(lambda x: calculate_sentence_rarity_log(x,freq_dict))
    gpt_model, gpt_tokenizer = load_gpt_model()
    df['ri'] = df['proc_sent'].progress_apply(lambda x: calculate_sentence_rarity_perplexity(x,gpt_model, gpt_tokenizer))

    print (df['ri'].describe())
    save_inference_to_file(df, load_path)

    # task_intervals = {'wnli':[0.04,0.05,0.06,0.08],'qqp':[0,0.1,0.25,0.6],
    #     'rte':[0.05,0.1,0.15,0.25]}

    task_intervals = {'wnli':[0.04,0.05,0.06,0.08],'qqp':[0,0.1,0.25,0.6],
        'rte':[26, 50, 120, 820], 'cola':[18,150,410,float('inf')]} #for gpt perplexity

    categorized_chunks = divide_into_chunks_by_RI(df,task_intervals[task])
    
    for i,chunk in enumerate(categorized_chunks):
        preds = chunk['preds'].tolist()
        gold_labels = chunk[gold_col[task]].tolist()
        accuracy = round(calculate_accuracy(preds, gold_labels, task),2)
        print (f'chunk {i}: --{task_intervals[task][i], task_intervals[task][i+1]}--  {len(preds)} sents\
            \n\t {accuracy} ')


def main():
    
    glue_tasks = ['wnli','qqp','rte','cola','sst',
            'mnli','qnli'][0:4] #mrpc,sts

    for task in glue_tasks:
        print ('\n','='*20,' ',task,' ','='*20,'\n')
        
        pipeline(task, dict(freq_sorted)
)


main()

