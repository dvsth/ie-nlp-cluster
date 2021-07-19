import transformers as tf
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

df = pickle.load(open('./data/clustering_df.pkl', 'rb'))

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

def embed_text(paragraph):
    encoded_input = tokenizer(paragraph, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        embedding = model(**encoded_input).last_hidden_state.mean(dim=(0,1)).numpy()
    return embedding.astype(np.float32)

df['embeddings'] = df.apply(lambda row: embed_text(row['paragraphs']), axis=1)
pickle.dump(df, open('./data/clusterviz_df.pkl', 'wb'))
