def get_roberta_params():
    
    return {
        'model_checkpoint' : "roberta-large",
        'source_column': 'gloss',
        'max_len': 150,
        'batch_size': 64,
        'dropout': 0.3,
        'learning_rate':1e-4,
        'num_epochs': 150,
        'early_stopping_limit': 5,
        'device': 'cuda:0',
        'loss_fn_name': 'cosine',
        'emb_type': 'fasttext',
        'use_adapters': True,
        'resume_from_checkpoint' : True,
        'output_size': 300,
        'knn_measure' : 'cosine',
        'num_warmup_steps': 300,
        'extra_data': False,
    }


def get_xlm_params():
    
    return {
        'model_checkpoint' : "xlm-roberta-large",
        'source_column': 'gloss',
        'max_len': 150,
        'batch_size': 16,
        'dropout': 0.1,
        'learning_rate':1e-4,
        'num_epochs': 150,
        'early_stopping_limit': 5,
        'device': 'cuda:0',
        'loss_fn_name': 'cosine',
        'emb_type': 'xlm-roberta-large',#'bert-base-uncased' 768,'fasttext-aligned' 300, fasttext 300, muse 300
        'use_adapters': True,
        'resume_from_checkpoint' : False,
        'output_size': 1024,
        'knn_measure' : 'cosine',
        'num_warmup_steps':300,
        'extra_data':False,
    }
