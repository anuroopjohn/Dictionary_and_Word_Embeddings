import torch
from transformers import AdamW, AutoTokenizer,  AutoModel

import torch
from transformers import AutoModel

class Model(torch.nn.Module):
    def __init__(self, model_checkpoint, output_size, dropout, device, loss_fn_name, use_adapters):
        super().__init__()
        self.base = AutoModel.from_pretrained(
                model_checkpoint
                )
        
        if use_adapters:
            self.base.add_adapter('adap')

            self.base.train_adapter("adap")
            self.base.set_active_adapters("adap")
        
        
        self.output_size = output_size
        self.device = device
        self.fc = torch.nn.Linear(self.base.config.hidden_size, self.output_size)
        
        
        self.dropout = torch.nn.Dropout(dropout)
        self.loss_fn_name = loss_fn_name
        assert loss_fn_name in ['l2', 'cosine', 'kl']
        if loss_fn_name == 'cosine':
            self.loss_fn = torch.nn.CosineEmbeddingLoss()
        elif  loss_fn_name == 'l2':
            self.loss_fn = torch.nn.MSELoss()
            
        elif loss_fn_name == 'kl':
            self.loss_fn = torch.nn.KLDivLoss(reduction= 'batchmean')#, log_target = True)
        
        else:
            print('Loss fn not right!')
    
    def forward(self, batch, mode = 'Train'):
        batch.to(self.device)
        #self.base.to(self.device)
        #self.fc.to(self.device)
        # actual = batch['act_emb']
        # del(batch['act_emb'])
        pools = self.base(input_ids=batch['input_ids'],
          attention_mask=batch['attention_mask'])['last_hidden_state'].mean(dim = 1)
        # print(hidden.shape)
        # print(pools.shape)
        
        out = self.fc(self.dropout(pools))
        
        if mode == 'Train':
            actual = batch['act_emb']

            if self.loss_fn_name == 'l2':
                loss = self.loss_fn(out, actual.float())
            elif self.loss_fn_name == 'cosine':#cosine
                loss = self.loss_fn(out,actual, torch.tensor([1], device = self.device))
                
            elif self.loss_fn_name == 'kl':
                loss = self.loss_fn(
                                    torch.nn.functional.log_softmax(out.double()),
                                    torch.nn.functional.softmax(actual.double())
                                    
                )

            else:
                print('Loss fn not right!')
            
            return loss, out, actual
        else:
            return out