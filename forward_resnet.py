from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf
import os
import numpy as np
import time
from embeddings_processing import *
import cv2
import argparse

from embeddings_processing_config import cfg

def resnet(dataDir):
	layers = 101

	sess = tf.Session()

	new_saver = tf.train.import_meta_graph('./models/'+meta_fn(layers))
	new_saver.restore(sess, './models/'+checkpoint_fn(layers))

	graph = tf.get_default_graph()
	prob_tensor_1 = graph.get_tensor_by_name("prob:0")
	prob_tensor_2 = graph.get_tensor_by_name("avg_pool:0")
	images = graph.get_tensor_by_name("images:0")
	for op in graph.get_operations():
	    print (op.name)

	print ("graph restored")

	start_time = time.time()

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
		
		feed_dict = {images: batch}	
		prob = sess.run([prob_tensor_1, prob_tensor_2], feed_dict=feed_dict)

		embedding = prob[0].flatten()
		if EMB1 is None:
			EMB1 = np.zeros((len(image_names), len(embedding)), dtype='float32')
		EMB1[i] = embedding

		embedding = prob[1].flatten()
		if EMB2 is None:
			EMB2 = np.zeros((len(image_names), len(embedding)), dtype='float32')
		EMB2[i] = embedding

		

	print("--- ResNet time: %s seconds ---" % (time.time() - start_time))

	return sess, images_list, image_names, EMB1, EMB2



if __name__ == '__main__':

	parser = argparse.ArgumentParser("Generates embeddings from network.")

	parser.add_argument("dataDir", type=str, help="folder with dataset images", nargs=1)
	parser.add_argument("--tb", action="store_true", help="genenrates tensorboard")

	args = parser.parse_args()	

	dataDir = args.dataDir[0]
	sess, images_list, image_names, EMB1, EMB2 = resnet(dataDir)
	
	print("Saving embeddings...")
	im_names = np.asarray(image_names)
    LOG_DIR_name = os.path.split(dataDir)
    np.save(os.path.join(LOG_DIR_name[0], 'image_names_' + LOG_DIR_name[1]), im_names)
    np.save(os.path.join(LOG_DIR_name[0], 'embeddings_' + LOG_DIR_name[1]), EMB)
	print("Done.")

	if args.tb:
		create_summary_embeddings(sess, images_list, image_names, EMB1, EMB2, 'tensorboard/test_' + LOG_DIR_name[1])
