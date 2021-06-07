import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch

sentences = pickle.load(open( "./data/sentences.pkl", "rb" ))
embeddings = pickle.load(open( "./data/embeddings.pkl", "rb" ))
years = pickle.load(open( "./data/years.pkl", "rb" ))

print(sentences.shape)
print(embeddings.shape)
print(years.shape)

final = []
for i in range(1):
  ans = []
  for j in range(len(embeddings)):
  	similarity = torch.cosine_similarity(embeddings[i].view(1,-1), 
  										 embeddings[j].view(1,-1)).item()
  	ans.append(similarity)
  final.append(ans)

top_k = 10
indices = np.argpartition(final, -top_k)[-top_k:]
indices = np.flip(indices[np.argsort(final[indices])])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('similarity rank')
ax.set_ylabel('year published')
ax.set_zlabel('similarity score')

for i in range(top_k):
  ax.scatter(i, years[indices[i]], final[indices[i]])

plt.show()
plt.savefig('viz.png')