import pandas as pd
import json
import numpy as np
import pickle

lang = 'en'
print (f'Loading dataframe for {lang}')
df = pd.read_csv(f"/data/users/kartik/thesis/data/full_dataset/{lang}.complete.csv")
df = df.reset_index()
print (df.head(2))

emb_types = ['sgns','char','electra']
for emb_type in emb_types:
    print (f"Converting {emb_type} embeddings from strings to numpy in dataframe...")
    df[emb_type] = df[emb_type].apply(lambda x: np.array(json.loads(x)))
    

db = []
cols = ['index']+emb_types
for i,group in df.groupby(['word','gloss']):
    item = {'word':i[0],'gloss':i[1]}
    for key in cols:
        item[key]=group[key].tolist()
    db.append(item)
print (f"Number of unique [word+gloss] : {len(db)}")

with open(f'../processed_data/{lang}.complete.wordgloss_group.pkl', 'wb') as outfile:
    pickle.dump(db, outfile)
