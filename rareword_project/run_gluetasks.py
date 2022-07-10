import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import random
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer


cache_dir = '/data/users/kartik/hfcache/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['TMPDIR'] = cache_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = '/data/users/kartik/Dictionary_and_Word_Embeddings/rareword_project/'


def load_data(task):
    
    sents = []
    if task =='qnli':
        with open(root_path+'/data/glue/QNLI/dev.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))

    elif task =='wnli':
        with open(root_path+'/data/glue/WNLI/dev.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))

    elif task == 'sst':
        with open(root_path+'/data/glue/SST-2/dev.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))
    
    elif task == 'cola':
        sents = [['id','label','orig','sentence']]
        with open(root_path+'/data/glue/CoLA/dev.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))

    elif task == 'mnli':
        with open(root_path+'/data/glue/MNLI/dev_matched.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))
    
    elif task == 'qqp':
        with open(root_path+'/data/glue/QQP/dev.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))
    
    elif task == 'sts':
        with open(root_path+'/data/glue/STS-B/dev.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))
    
    elif task == 'rte':
        with open(root_path+'/data/glue/RTE/dev.tsv') as f:
            for line in f:
                sents.append(line.strip().split('\t'))
                
    df = pd.DataFrame(sents[1:],columns=sents[0])
    print (f'Task: {task}, Num sents: {len(df)}')
    return df

def prepare_bert_finetuned_model(task):
    
    # model_checkpoint = 'bert-base-uncased'
    if task =='qnli':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-qnli'
    elif task=='sst':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-sst2'
    elif task=='wnli':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-wnli'
    elif task=='mnli':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-mnli'
    elif task=='mrpc':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-mrpc'
    elif task=='qqp':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-qqp'
    elif task=='cola':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-cola'
    elif task=='sts':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-stsb'
    elif task=='rte':
        model_checkpoint = 'gchhablani/bert-base-cased-finetuned-rte'

    print ('Loading HF checkpoint: ', model_checkpoint)
    ft_model = BertForSequenceClassification.from_pretrained(model_checkpoint)
    ft_tokenizers = BertTokenizer.from_pretrained(model_checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ft_model = ft_model.to(device)
    return ft_model, ft_tokenizers

 
def run_nli_inference(questions, sentences, itos, ft_model, ft_tokenizers):
   
    preds= []
    confidence = []

    for ridx in tqdm(range(len(sentences))):
        sentence1 = questions[ridx]
        sentence2 = sentences[ridx]

        # print (f'sentence1: {sentence1}, sentence2: {sentence2}')

        tokenized_sent = ft_tokenizers(sentence1, sentence2, return_tensors="pt")
        tokenized_sent = tokenized_sent.to(device)
        logits = ft_model(**tokenized_sent).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]

        result = torch.argmax(logits, dim=1).tolist()[0]
        prob = probs[result]
        result = itos[result]

        preds.append(result)
        confidence.append(prob)

    return preds,confidence

def run_sst_inference(sentences, itos, ft_model, ft_tokenizers):
    ''' 
    Stanford Sentiment Task in GLUE
    '''
    preds= []
    confidence = []

    for ridx in tqdm(range(len(sentences))):
        sentence = sentences[ridx]

        # print (f'sentence1: {sentence1}, sentence2: {sentence2}')

        tokenized_sent = ft_tokenizers(sentence, return_tensors="pt")
        tokenized_sent = tokenized_sent.to(device)
        logits = ft_model(**tokenized_sent).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]

        result = torch.argmax(logits, dim=1).tolist()[0]
        prob = probs[result]
        result = itos[result]

        preds.append(result)
        confidence.append(prob)

    return preds,confidence

def save_inference_to_file(df, path):
    
    df.to_csv(path, index=False)

def calculate_accuracy(preds, golds, task):
    
    assert len(golds)==len(preds)

    if task == 'mnli':
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(golds, preds)
    else:
        accuracy = [preds[i]==golds[i] for i in range(len(preds))]
        return sum(accuracy)/len(accuracy)


def run_task(task, df, model, tokenizer):
    
    print ('Running Inference....')

    if task =='qnli':
        itos = ['entailment','not_entailment']
        qns, sents, gold_labels = df['question'].tolist(), df['sentence'].tolist(), df['label'].tolist()
        preds, confidence = run_nli_inference(qns, sents, itos, model, tokenizer)

    elif task == 'sst':
        itos = ['0','1']
        sents, gold_labels = df['sentence'].tolist(), df['label'].tolist()
        preds, confidence = run_sst_inference(sents, itos, model, tokenizer)
    
    elif task == 'cola':
        itos = ['0','1']
        sents, gold_labels = df['sentence'].tolist(), df['label'].tolist()
        preds, confidence = run_sst_inference(sents, itos, model, tokenizer)
    
    if task =='wnli':
        itos = ['0','1']
        sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['label'].tolist()
        preds, confidence = run_nli_inference(sents1, sents2, itos, model, tokenizer)
   
    if task =='mnli':
        itos = ['neutral','contradiction','entailment']
        stoi = {k:v for v,k in enumerate(itos)}

        sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['gold_label'].tolist()
        preds, confidence = run_nli_inference(sents1, sents2, itos, model, tokenizer)
        
        gold_labels = [stoi[i] for i in gold_labels]
        preds = [stoi[i] for i in preds]

    if task =='qqp':
        itos = ['0','1']
        sents1, sents2, gold_labels = df['question1'].tolist(), df['question2'].tolist(), df['is_duplicate'].tolist()
        preds, confidence = run_nli_inference(sents1, sents2, itos, model, tokenizer)
        
    if task =='rte':
        itos = ['entailment','not_entailment']
        sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['label'].tolist()
        preds, confidence = run_nli_inference(sents1, sents2, itos, model, tokenizer)
   
    # if task =='sts':
    #     itos = ['entailment','not_entailment']
    #     sents1, sents2, gold_labels = df['sentence1'].tolist(), df['sentence2'].tolist(), df['label'].tolist()
    #     preds, confidence = run_nli_inference(sents1, sents2, itos, model, tokenizer)

    df['preds'] = preds
    df['confidence'] = confidence

    save_path = f'./glue_preds/{task}_bertft.csv'
    accuracy = calculate_accuracy(preds, gold_labels, task)
    print ('Accuracy: ',accuracy)

    save_inference_to_file(df, save_path)


def main():
    
    glue_tasks = ['rte','wnli','cola','sst','qqp',
            'mnli','qnli'] #mrpc,sts

    for task in glue_tasks:
        print ('\n','='*20,' ',task,' ','='*20,'\n')
        
        df = load_data(task)
        print (df.head(3))
        

        model, tokenizer = prepare_bert_finetuned_model(task)

        run_task(task, df, model, tokenizer)


main()








