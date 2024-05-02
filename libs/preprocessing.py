
import dataset
from sklearn.decomposition import PCA
import numpy as np

def pca_features(dataset, components):

    (samples,features,frames) = dataset.shape
    dataset_reshaped = np.zeros((samples, components, frames))

    for i in range(0,frames):
        frame = dataset[:,:,i]
        pca = PCA(n_components=components)
        frame_pca = pca.fit_transform(frame)
        dataset_pca[:,:,i] = frame_pca
    
    return dataset_pca





    
