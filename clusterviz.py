from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors
import pickle
import numpy as np

df = pickle.load(open('./clusterviz_df.pkl', 'rb'))

tsne = TSNE(random_state=1, n_iter=1000, metric="cosine", n_components=2)
arr = df['embeddings'].to_numpy()
X = np.vstack(arr[:]).astype(np.float)
print(X.shape, X.dtype)

embs = tsne.fit_transform(X)
# Add to dataframe for convenience
df['x'] = embs[:, 0]
df['y'] = embs[:, 1]

fig, ax = plt.subplots()
ax.scatter(df['x'], df['y'])

annotations = df['abstract'].to_numpy()
mplcursors.cursor(ax, hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(annotations[sel.target.index]))
    
plt.show()
