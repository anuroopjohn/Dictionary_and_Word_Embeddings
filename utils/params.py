
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
        'knn_measure' : 'cosine'
    }


def get_xlm_params():
    
    return {
        'model_checkpoint' : "xlm-roberta-large",
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
        'resume_from_checkpoint' : False,
        'output_size': 300,
        'knn_measure' : 'cosine'
    }



 #
