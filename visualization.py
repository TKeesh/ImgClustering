from visualization_config import cfg

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc
import numpy as np

from glob import glob
from scipy import spatial
import os

from synset import *
#from tsne import tsne

def create_summary_embeddings(sess, images, image_names, EMB, LOG_DIR):
    """
    Create summary for embeddings.
    :param sess: Session object.
    :param images: Images.
    :param image_names: Image names.
    :param EMB: Embeddings.
    :param LOG_DIR: Tensorboard dir.
    :return:
    """
    if not len(LOG_DIR): LOG_DIR = cfg.TENSORBOARD_PATH
    # create summary writer
    test_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    # The embedding variable, which needs to be stored
    # Note this must a Variable not a Tensor!
    embedding_var = tf.Variable(EMB, name='output_tensor')
    # init
    sess.run(embedding_var.initializer)
    #create summary writter
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    # create config
    config = projector.ProjectorConfig()
    # create embedding
    embedding = config.embeddings.add()
    # define name
    embedding.tensor_name = embedding_var.name
    # define metadata file
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    # config sprite
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png')
    embedding.sprite.single_image_dim.extend([cfg.EMB_IMAGE_HEIGHT, cfg.EMB_IMAGE_WIDTH])

    #NASE save embedings as np
    im_names = np.asarray(image_names)
    np.save('image_names', im_names)
    np.save('embeddings', EMB)

    # projector run
    projector.visualize_embeddings(summary_writer, config)

    # save embeddings
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), 1)

    # write metadata
    if len(EMB[0]) == 1000:
        metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
        metadata_file.write('Name\tClass\n')
        cnf = open('./classes.txt', 'w')
        for i, name in enumerate(image_names):
            prob = EMB[i]
            pred = np.argsort(prob)[::-1]
            metadata_file.write('%06d\t%s\n' % (i, name+': '+' '.join(synset[pred[0]].split()[1:])))
            cnf.write(name + ': ')
            topX = [' '.join(synset[pred[i]].split()[1:]) for i in range(7)]
            print (topX)
            cnf.write(' | '.join(topX))            
            cnf.write('\n')
        cnf.close()
        metadata_file.close()
    else:
        metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
        metadata_file.write('Name\tClass\n')
        for i, name in enumerate(image_names):
            metadata_file.write('%06d\t%s\n' % (i, name))
        metadata_file.close()

    print('embeddings saved')

    # create sprite
    print('creating sprite')
    if len(images): sprite = _images_to_sprite(images)
    print('saving sprite')
    if len(images): scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)
    # with open('names.txt', 'w') as f:
    #     f.write('\n'.join(image_names))


def _images_to_sprite(data):
    """
    Creates the sprite image along with any necessary padding.
    :param data: NxHxW[x3] tensor containing the images.
    :return: Properly shaped HxWx3 image with any necessary padding.
    """

    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)

    pom = np.zeros(data.shape)
    pom[:, :, 0] = data[:, :, 2]
    pom[:, :, 1] = data[:, :, 1]
    pom[:, :, 2] = data[:, :, 0]
    return pom


def make_folders(clusters, datasetFolder, extension, fnames):
    folder = datasetFolder + extension
    if os.path.exists(folder): 
        import shutil
        shutil.rmtree(folder)
    for imgi in range(len(clusters)):
        # print (fnames[imgi])
        if not os.path.exists(folder + '\\' + str(clusters[imgi])):
            os.makedirs(folder + '\\' + str(clusters[imgi]))
        imgorg = glob(datasetFolder + '/' + fnames[imgi].split('.')[0] + '.*')[0]
        try: img = (scipy.misc.imread(imgorg)[:,:,:3]).astype('float32')
        except: img = (scipy.misc.imread(imgorg)[:]).astype('float32')
        scipy.misc.imsave(os.path.join(folder + '\\' + str(clusters[imgi]) + '\\', imgorg.split('\\')[-1]), img)


def create_folders(EMB, image_names = '', images_folder = ''):    
    from sklearn.cluster import DBSCAN   
    from sklearn.manifold import TSNE
    

    model = TSNE(init='pca', n_components=3, random_state=0, n_iter=800, perplexity=5, learning_rate=10, metric='cosine', method='exact', n_iter_without_progress=1000)    
    np.set_printoptions(suppress=True)
    print("TSNEfit")    
    mat2D = model.fit_transform(EMB) 
    # print(type(EMB))
    # print(EMB.dtype)
    # print (EMB)
    # try:
    #     input("Press enter to continue")
    # except SyntaxError:
    #     pass
    #mat2D = tsne(X = EMB.astype('float64'), no_dims = 3, initial_dims = len(EMB[0]), perplexity = 5.0)
    # print(type(mat2D))
    # print(mat2D.dtype)
    # print (mat2D)
    
    print(mat2D.shape)
    maxx = np.max(mat2D[:, 0])
    minx = np.min(mat2D[:, 0])

    maxy = np.max(mat2D[:, 1])
    miny = np.min(mat2D[:, 1])

    maxz = np.max(mat2D[:, 2])
    minz = np.min(mat2D[:, 2])

    print("maxx = " + str(maxx))
    print("minx = " + str(minx))
    print("maxy = " + str(maxy))
    print("miny = " + str(miny))
    print("maxz = " + str(maxz))
    print("minz = " + str(minz))
    # eps = np.maximum(np.maximum(maxx-minx, maxy-miny), maxz-minz) / 10.0
    eps = ((maxx-minx) + (maxy-miny) + (maxz-minz)) / 3.0 * 490000 / len(EMB) / len(EMB) / len(EMB)
    print ('eps = ', eps)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # print(mat2D[:,0])
    ax.scatter(mat2D[:,0], mat2D[:,1], mat2D[:,2], c='r')
    #plt.show()
    fig.savefig('./TSNEplot.png')
    #mat2D = mat2D.transpose()

    #quit()
    
    # print (type(mat2D), mat2D.shape)
    clusters = DBSCAN(eps=eps, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    #clusters = DBSCAN(eps=0.75, algorithm='brute', min_samples=1, metric='cosine').fit_predict(EMB)
    # print(clusters)

    if len(images_folder): 
        datasetFolder = images_folder
    else: 
        datasetFolder = cfg.TRAIN_FOLDER
    if len(image_names): 
        fnames = image_names
    else: 
        fnames = os.listdir(images_folder)
        fnames.sort()
    try: 
        fnames = [x.decode('UTF-8') for x in fnames]
    except: pass
    # print (fnames[0], type(fnames))
    datasetFolder = datasetFolder.strip('.').strip('/')
    outFile = open(datasetFolder+'.txt','w')
    for imgi in range(len(EMB)):
        outFile.write('\n')
        outFile.write(fnames[imgi] + '\n')
        outFile.write('Cluster: ' + str(clusters[imgi]) + '\n')
    outFile.close()
  
    # if os.path.exists(datasetFolder+'Clusters'): 
    #     import shutil
    #     shutil.rmtree(datasetFolder+'Clusters')
    # for imgi in range(len(EMB)):
    #     print (fnames[imgi])
    #     if not os.path.exists(datasetFolder+'Clusters' + '\\' + str(clusters[imgi])):
    #         os.makedirs(datasetFolder+'Clusters' + '\\' + str(clusters[imgi]))
    #     imgorg = glob(datasetFolder + '/' + fnames[imgi].split('.')[0] + '.*')[0]
    #     try: img = (scipy.misc.imread(imgorg)[:,:,:3]).astype('float32')
    #     except: img = (scipy.misc.imread(imgorg)[:]).astype('float32')
    #     scipy.misc.imsave(os.path.join(datasetFolder+'Clusters' + '\\' + str(clusters[imgi]) + '\\', imgorg.split('\\')[-1]), img)
    make_folders(clusters, datasetFolder, 'Clusters', fnames)

    clusters_mean = np.zeros((max(clusters)+1, len(EMB[0])))
    clusters_examples = np.zeros((max(clusters)+1, 1))
    for i, c in enumerate(clusters):
        clusters_mean[c] += EMB[i]
        clusters_examples += 1
    for i in range(len(clusters_mean)):
        clusters_mean[i] /= clusters_examples[i]


    # result = 1 - spatial.distance.cosine(EMB[1], EMB[3])
    # print("kosinusna_1008: " + str(result))
    # result = np.linalg.norm(EMB[1]-EMB[3])
    # print("euklidna_1008: " + str(result))

    # result = 1 - spatial.distance.cosine(mat2D[1], mat2D[3])
    # print("kosinusna_3: " + str(result))
    # result = np.linalg.norm(mat2D[1]-mat2D[3])
    # print("euklidna_3: " + str(result))

    model = TSNE(n_components=3, random_state=0, n_iter=720, perplexity=5, learning_rate=10, metric='cosine')
    print (clusters_mean.shape)
    mat2D = model.fit_transform(clusters_mean) 

    clusters_of_clusters = DBSCAN(eps=1.5, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # print(clusters_of_clusters)
    clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    make_folders(clusters_of_clusters, datasetFolder, '1_5', fnames)
    clusters_of_clusters = DBSCAN(eps=2, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # print(clusters_of_clusters)
    clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    make_folders(clusters_of_clusters, datasetFolder, '2', fnames)
    clusters_of_clusters = DBSCAN(eps=2.5, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # print(clusters_of_clusters)
    clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    make_folders(clusters_of_clusters, datasetFolder, '2_5', fnames)
    clusters_of_clusters = DBSCAN(eps=3, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # print(clusters_of_clusters)
    clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    make_folders(clusters_of_clusters, datasetFolder, '3', fnames)
'''
    for i in range(len(EMB[1])):
        p = 1.0/(1+np.exp(EMB[1][i]))
        print('*'*round(p*20))
        p = 1.0/(1+np.exp(EMB[3][i]))
        print('*'*round(p*20))
        print()
'''


    # for x in EMB[1]:
    #     print(1.0/(1+np.exp(x)))
    # for x in EMB[3]:
    #     print(1.0/(1+np.exp(x)))
    # DBSCAN(cosine), nad clusters_mend(EMB), nakon TSNE-DBSCAN(euclidean) ne valja !!!

    # clusters_of_clusters = DBSCAN(eps=0.1, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
    # print(clusters_of_clusters)
    # clusters_of_clusters = DBSCAN(eps=0.2, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
    # print(clusters_of_clusters)
    # clusters_of_clusters = DBSCAN(eps=0.3, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
    # print(clusters_of_clusters)
    # clusters_of_clusters = DBSCAN(eps=0.4, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
    # print(clusters_of_clusters)
    # clusters_of_clusters = DBSCAN(eps=0.5, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
    # print(clusters_of_clusters)
    # clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    # make_folders(clusters_of_clusters, datasetFolder, '0_5', fnames)
    # clusters_of_clusters = DBSCAN(eps=0.6, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
    # print(clusters_of_clusters)
    # clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    # make_folders(clusters_of_clusters, datasetFolder, '0_6', fnames)
    # clusters_of_clusters = DBSCAN(eps=0.7, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
    # print(clusters_of_clusters)
    # clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    # make_folders(clusters_of_clusters, datasetFolder, '0_7', fnames)
