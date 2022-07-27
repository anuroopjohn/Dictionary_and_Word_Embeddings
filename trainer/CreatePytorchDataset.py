
import numpy as np

####################################
#   Create Pytorch Datasets
####################################

class TrainDataset:
    def __init__(self, train, tokenizer, source_column, max_length, emb_type):
        self.train = train
        self.tokenizer = tokenizer
        self.source_column = source_column
        self.emb_column = emb_type
        self.max_length = max_length
        self.transform = True
    def __len__(self):
        return len(self.train)
        
        
    def __getitem__(self, idx):
        input_text = self.train[self.source_column].loc[idx]
        
        
        
        tokenized_source_text = self.tokenizer(input_text, truncation=True, padding=True, max_length = self.max_length)
        actual_embs = self.train.loc[idx][self.emb_column]
        tokenized_source_text['act_emb'] = actual_embs#actual_embs[0]
        return tokenized_source_text
    
class ValDataset:
    def __init__(self, val, tokenizer, source_column, max_length, emb_type):
        self.val = val
        self.tokenizer = tokenizer
        self.source_column = source_column
        self.emb_column = emb_type
        self.max_length = max_length
    
    def __len__(self):
        return len(self.val)
        
        
    def __getitem__(self, idx):
        input_text = self.val[self.source_column].loc[idx]
        
        
            
        tokenized_source_text = self.tokenizer(input_text, truncation=True, padding=True, max_length = self.max_length)
        actual_embs = self.val.loc[idx][self.emb_column]
        tokenized_source_text['act_emb'] = actual_embs#actual_embs[0]
        return tokenized_source_text
    
    
