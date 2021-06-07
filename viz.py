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

i = 1000
ans = []
final = []
for j in range(len(embeddings)):
  similarity = torch.cosine_similarity(embeddings[i].view(1,-1), 
                      embeddings[j].view(1,-1)).item()
  ans.append(similarity)
final.append(ans)

final = np.asarray(final[0])
top_k = 50
indices = np.argpartition(final, -top_k)[-top_k:]
print(indices)
print(final[indices])
print(np.argsort(final[indices]))
indices = np.flip(indices[np.argsort(final[indices])])
indices = indices[1:]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('similarity rank')
ax.set_ylabel('year published')
ax.set_zlabel('similarity score')

for i in range(top_k - 1):
  ax.scatter(i, years[indices[i]], final[indices[i]])

plt.show()
plt.savefig('viz.png')