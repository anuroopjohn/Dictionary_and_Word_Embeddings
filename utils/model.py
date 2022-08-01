import torch
from transformers import AdamW, AutoTokenizer,  AutoModel

import torch
from transformers import AutoModel

class Model(torch.nn.Module):
    def __init__(self, model_checkpoint, output_size, dropout, device, loss_fn_name, use_adapters):
        super().__init__()
        self.base = AutoModel.from_pretrained(
                model_checkpoint, output_hidden_states=True
                )
        
        if use_adapters:
            self.base.add_adapter('adap')

            self.base.train_adapter("adap")
            self.base.set_active_adapters("adap")
        
        
        self.output_size = output_size
        self.device = device
        
        # self.layer_weights = torch.nn.Linear(25, 1)

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
    
    def masked_mean(self, t, mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s/d

    def forward(self, batch, mode = 'Train'):
        batch.to(self.device)

        pools = self.base(input_ids=batch['input_ids'],
          attention_mask=batch['attention_mask'])  #shape: bs,max_len,1024 num_layers in xlmrL:25
        # batch['attention_mask'] shape: bs, max_len
        
        pools = pools['last_hidden_state']
        # pools = torch.stack(pools['hidden_states']).permute(1,2,3,0)
        # pools = pools['hidden_states'][20]

        # pools = self.layer_weights(pools).squeeze(-1)
        # print (pools.shape)
        # pools = pools['hidden_states'].mean()

        pools = self.masked_mean(pools,batch['attention_mask'])
        # pools = pools['last_hidden_state'].mean(dim = 1)

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