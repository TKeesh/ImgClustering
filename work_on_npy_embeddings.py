import numpy as np
from visualization import create_folders, create_folders_2
# from sklearn import mixture

EMB = np.load('./embeddings_2048_ResNet_101L.npy')[:]
image_names = np.load('./image_names_2048_ResNet_101L.npy')[:]

print (EMB)
# #create_summary_embeddings(sess, images_list, image_names, EMB, '')

#print(EMB.shape)
for i in [1000,2000,3000,4000,5000,6000]:
	create_folders(EMB[:i], image_names[:i], './datasetJPG/')
create_folders(EMB[:], image_names[:], './datasetJPG/') # za sve slike 


# print('fitting...')
# dpgmm = mixture.BayesianGaussianMixture(n_components=15, covariance_type='full').fit(EMB)
# print('predicting...')
# pred = dpgmm.predict(EMB)
# print(len(pred))
# print(pred)
# print('done')
