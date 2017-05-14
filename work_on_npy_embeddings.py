import numpy as np
from visualization import create_folders
# from sklearn import mixture

EMB = np.load('./tensorboard/embeddings_and_tensorboards/embeddings_2048_ResNet_101L.npy')[:2000]
image_names = np.load('./tensorboard/embeddings_and_tensorboards/image_names_2048_ResNet_101L.npy')[:2000]

print (EMB)
# #create_summary_embeddings(sess, images_list, image_names, EMB, '')
create_folders(EMB, image_names, './datasetJPG/')

# print('fitting...')
# dpgmm = mixture.BayesianGaussianMixture(n_components=15, covariance_type='full').fit(EMB)
# print('predicting...')
# pred = dpgmm.predict(EMB)
# print(len(pred))
# print(pred)
# print('done')
