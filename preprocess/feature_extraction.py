from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings("ignore")

# model_checkpoint = ['xlm-roberta-base', 'distilbert-base-uncased','bert-base-uncased','albert-base-v2'][2]


class FeatureExtractor:

    def __init__(self, model_checkpoint):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, output_hidden_states = True)
        self.model = AutoModel.from_pretrained(model_checkpoint,output_attentions=False,output_hidden_states=False)


    def extract_feature(self, sentence, word):
        '''
            tokenize sentence and extract contextual feature of a word from a sentence
        '''

        sentence = "[CLS] " + sentence.lower() + " [SEP]" 

        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
        sent_token_ids = self.tokenizer.encode_plus(sentence, add_special_tokens=True, 
                                return_attention_mask=True, pad_to_max_length=True, 
                                max_length=256, return_tensors='pt')['input_ids']
        
        with torch.no_grad():
            sent_feature = self.model(sent_token_ids)['last_hidden_state']
        id_ = self.tokenizer.vocab.get(word,None)

        if id_ == None:
            # word_tokens  = self.tokenizer.tokenize(targets[i]) ### gooder --> good, ##er
            word_token_ids = self.tokenizer.encode_plus(word)['input_ids'][1:-1]
            id_ = word_token_ids[0]

        for idx,tok_id in enumerate(sent_token_ids[0]):
            if tok_id == id_:
                token_feature = sent_feature[0,idx,:]
                break
        
        return token_feature

