lang = 'en'
root = '/data/users/kartik/Dictionary_and_Word_Embeddings/data/'


raw_paths = {}
raw_paths['full_data'] =f'{root}/data/full_dataset/{lang}.complete.csv'
raw_paths['train'] =f'{root}/data/train_data/{lang}.dev.json'
raw_paths['dev'] =f'{root}/data/train_data/{lang}.dev.json'
raw_paths['test'] =f'{root}/data/test_data/{lang}.test.revdict.json'


proc_paths = {'codwoe':{},'wikt':{}}
proc_paths['full_data'] = f'{root}/processed_data/{lang}.complete.wordgloss_group.pkl'


proc_paths['codwoe']['train'] = f'{root}/processed_data/codwoe.{lang}.train.wordgloss.pkl'
proc_paths['codwoe']['dev'] = f'{root}/processed_data/codwoe.{lang}.dev.wordgloss.pkl'
proc_paths['codwoe']['test'] = f'{root}/processed_data/codwoe.{lang}.test.wordgloss.pkl'

proc_paths['wikt']['train'] = f'{root}/processed_data/wikt.{lang}.train.wordgloss.pkl'
proc_paths['wikt']['dev'] = f'{root}/processed_data/wikt.{lang}.dev.wordgloss.pkl'
proc_paths['wikt']['test'] = f'{root}/processed_data/wikt.{lang}.test.wordgloss.pkl'

sampled_data_path = f'{root}/processed_data/{lang}.10k_sampled.wordgloss_group.pkl'

sampled_rougescore_path = f'{root}/processed_data/{lang}.10k_sampled.pairwise_rouge.pkl'
