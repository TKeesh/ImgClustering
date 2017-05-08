from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf
import os
import numpy as np
import time

layers = 50

sess = tf.Session()

new_saver = tf.train.import_meta_graph('./models/'+meta_fn(layers))
new_saver.restore(sess, './models/'+checkpoint_fn(layers))

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
for op in graph.get_operations():
    print (op.name)

#init = tf.initialize_all_variables()
#sess.run(init)
print ("graph restored")

start_time = time.time()

image_batch = []
image_list = os.listdir('./data/')
for img in image_list[:10]:
	print(img)
	img = load_image("./data/{0}".format(img))
	batch = img.reshape((1, 224, 224, 3))
	#image_batch.append(batch)
	
	feed_dict = {images: batch}	
	prob = sess.run(prob_tensor, feed_dict=feed_dict)

	

# image_batch = np.array(image_batch)
# feed_dict = {images: image_batch}
# prob = sess.run(prob_tensor, feed_dict=feed_dict)

print("--- %s seconds ---" % (time.time() - start_time))

print_prob(prob[0])
print()
