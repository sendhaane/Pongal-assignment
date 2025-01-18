from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
data = pd.DataFrame(np.random.rand(1000, 5), columns=['Speed', 'Traffic', 'Weather', 'Lighting', 'Road_Type'])


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Reduced Data Shape:", reduced_data.shape)
