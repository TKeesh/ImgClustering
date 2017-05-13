from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf
import os
import numpy as np
import time
from visualization import *
import cv2

layers = 101

dataDir = './data/'
#dataDir = './dataOznacena/'

sess = tf.Session()

new_saver = tf.train.import_meta_graph('./models/'+meta_fn(layers))
new_saver.restore(sess, './models/'+checkpoint_fn(layers))

graph = tf.get_default_graph()
prob_tensor_1 = graph.get_tensor_by_name("prob:0")
prob_tensor_2 = graph.get_tensor_by_name("avg_pool:0")
images = graph.get_tensor_by_name("images:0")
for op in graph.get_operations():
    print (op.name)

#init = tf.initialize_all_variables()
#sess.run(init)
print ("graph restored")

start_time = time.time()

#image_batch = []
EMB1, EMB2 = None, None
image_names = os.listdir(dataDir)
image_names = image_names[:]
images_list = np.zeros(shape=(len(image_names), cfg.EMB_IMAGE_HEIGHT, cfg.EMB_IMAGE_WIDTH, 3))
for i, img_name in enumerate(image_names):
	if i == 1:
		start_time = time.time()
	print(str(i+1).rjust(4) + '/' + str(len(image_names)) + ' - ' + img_name)
	img = load_image(dataDir + img_name)

	image = cv2.imread(dataDir + img_name)
	image = cv2.resize(image, (cfg.EMB_IMAGE_HEIGHT, cfg.EMB_IMAGE_WIDTH), interpolation = cv2.INTER_CUBIC)
	images_list[i] = image

	batch = img.reshape((1, 224, 224, 3))
	#image_batch.append(batch)
	
	feed_dict = {images: batch}	
	prob = sess.run([prob_tensor_1, prob_tensor_2], feed_dict=feed_dict)
	#print_prob(prob[0])
		
	# treba li ovaj flatten?! Treba!
	embedding = prob[0].flatten()
	if EMB1 is None:
		EMB1 = np.zeros((len(image_names), len(embedding)), dtype='float32')
	EMB1[i] = embedding

	embedding = prob[1].flatten()
	if EMB2 is None:
		EMB2 = np.zeros((len(image_names), len(embedding)), dtype='float32')
	EMB2[i] = embedding

	

# image_batch = np.array(image_batch)
# feed_dict = {images: image_batch}
# prob = sess.run(prob_tensor, feed_dict=feed_dict)

print("--- %s seconds ---" % (time.time() - start_time))

print('saving embeddings')
# args --tb
#create_summary_embeddings(sess, images_list, image_names, EMB1, 'tensorboard/test_' + str(len(EMB1[0])) + '_ResNet-L' + str(layers)) 
create_summary_embeddings(sess, images_list, image_names, EMB2, 'tensorboard/test_' + str(len(EMB2[0])) + '_ResNet-L' + str(layers))
print('done')
print('creating folders')
# args --f
create_folders(EMB2, image_names, dataDir)
print('done')
