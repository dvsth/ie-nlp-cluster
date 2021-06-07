import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('./data/papers.csv')

df = df[ df['abstract'] != 'Abstract Missing']
df.reset_index(drop=True, inplace=True)

years = np.array(df.year)
sentences = np.array(df.abstract)

pickle.dump(sentences, open( "./data/sentences.pkl", "wb" ))
pickle.dump(years, open( "./data/years.pkl", "wb" ))