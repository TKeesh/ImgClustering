import numpy as np
from visualization import create_folders


EMB = np.load('./embeddings.npy')[:50]
image_names = np.load('./image_names.npy')[:50]



# print('saving embeddings')
# #create_summary_embeddings(sess, images_list, image_names, EMB1, '')
# create_summary_embeddings(sess, images_list, image_names, EMB2, '')
# print('done')
# print('creating folders')
create_folders(EMB, image_names, './data/')
print('done')