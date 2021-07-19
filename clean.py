import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('./data/papers.csv')

df = df[df['abstract'] != 'Abstract Missing']
df = df[df['abstract'].notna()]
df['abstract'] = df['abstract'].str.replace('\r\n', '')
df['abstract'] = df['abstract'].str.replace('\x03', '.')
df['abstract'] = df['abstract'].str.replace('i.e.', 'ie')
df['abstract'] = df['abstract'].str.replace('e.g.', 'example')
df['abstract'] = df['abstract'].str.replace(' ex.', ' example')

df.reset_index(drop=True, inplace=True)

years = np.array(df.year)
sentences = np.array(df.abstract)

pickle.dump(sentences, open("./data/sentences.pkl", "wb"))
pickle.dump(years, open("./data/years.pkl", "wb"))

def str_to_arr(row):
    out = row['abstract'].split(".")
    out = [x.strip() for x in out]
    out = list(filter(lambda x: len(x) > 0, out))
    return out

df["paragraphs"] = df.apply(str_to_arr, axis=1)
df = df[["paragraphs", "abstract"]]
pickle.dump(df, open('./data/clustering_df.pkl', 'wb'))