from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import numpy as np
with open('C:/Users/bedis/Desktop/pkl1/L.pkl', 'rb') as f:
   L = pickle.load(f)
print(L)
pca2 = PCA(n_components=795)

x=pca2.fit(L[:100000])

plt.plot(np.cumsum(pca2.explained_variance_ratio_))


with open('C:/Users/bedis/Desktop/pkl1/PCA.pkl', 'wb') as f:
  mynewlist = pickle.dump(pca2,f)
plt.xlabel('number of Components')
plt.ylabel('Explained Variances')
plt.show()
