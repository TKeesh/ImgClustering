from embeddings_processing_config import cfg

from scipy import spatial
import scipy.misc
import numpy as np

import os, time, argparse

precision_boost = False


def create_summary_embeddings(sess, images, image_names, EMB1, EMB2, LOG_DIR):

    import synset
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector

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
    embedding_var = tf.Variable(EMB2, name='output_tensor')
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
    # im_names = np.asarray(image_names)
    # LOG_DIR_name = os.path.split(LOG_DIR)
    # np.save(os.path.join(LOG_DIR_name[0], 'image_names_' + LOG_DIR_name[1]), im_names)
    # np.save(os.path.join(LOG_DIR_name[0], 'embeddings_' + LOG_DIR_name[1]), EMB2)

    # projector run
    projector.visualize_embeddings(summary_writer, config)

    # save embeddings
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), 1)

    # write metadata
    metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
    metadata_file.write('Name\tClass\n')
    cnf = open(os.path.join(LOG_DIR_name[0], 'classes.txt'), 'w')
    for i, name in enumerate(image_names):
        prob = EMB1[i]
        pred = np.argsort(prob)[::-1]
        metadata_file.write('%06d\t%s\n' % (i, name+': '+' '.join(synset[pred[0]].split()[1:])))
        cnf.write(name + ': ')
        topX = [' '.join(synset[pred[i]].split()[1:]) for i in range(7)]
        cnf.write(' | '.join(topX))            
        cnf.write('\n')
    cnf.close()
    metadata_file.close()


    print('embeddings saved')

    # create sprite
    print('creating sprite')
    if len(images): sprite = _images_to_sprite(images)
    print('saving sprite')
    if len(images): scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)


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
    from glob import glob

    folder = './' + os.path.split(datasetFolder.strip('/').strip('\\').strip('\\\\').strip('//'))[-1] + extension
    print ('Creating folders with clusters ({0})...'.format(folder))
    # datasetFolder = datasetFolder.strip('.').strip('/')
    # outFile = open(datasetFolder+'.txt','w')
    # for imgi in range(len(fnames)):
    #     outFile.write('\n')
    #     outFile.write(fnames[imgi] + '\n')
    #     outFile.write('Cluster: ' + str(clusters[imgi]) + '\n')
    # outFile.close()
    if os.path.exists(folder): 
        import shutil
        shutil.rmtree(folder)
    for imgi in range(len(clusters)):
        # print (fnames[imgi])
        out_folder = os.path.join(folder, str(clusters[imgi]))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        imgorg = glob(os.path.join(datasetFolder, fnames[imgi].split('.')[0]) + '.*')[0]
        try: img = (scipy.misc.imread(imgorg)[:,:,:3]).astype('float32')
        except: img = (scipy.misc.imread(imgorg)[:]).astype('float32')
        # print(os.path.join(out_folder, os.path.split(imgorg)[-1]))
        # print(img.shape)
        # os.chdir(os.path.dirname(__file__))
        scipy.misc.imsave(os.path.join(out_folder, os.path.split(imgorg)[-1]), img)
    print ('Folders created')


def connections_valid(clusters_of_clusters, clusters_mean):
    import operator

    connected = {}
    for i in range(len(clusters_mean)):
        avg_dist, br = 0, 0
        for j in range(len(clusters_mean)):
            if j <= i: continue
            # print (i, j)
            # result = spatial.distance.cosine(clusters_mean[i], clusters_mean[j])
            # print("kosinusna_1008: " + str(result))
            if clusters_of_clusters[i] == clusters_of_clusters[j]:
                #connected.append([i, j, spatial.distance.cosine(clusters_mean[i], clusters_mean[j])]) 
                avg_dist += spatial.distance.cosine(clusters_mean[i], clusters_mean[j])
                br += 1
        if br > 0:
            avg_dist /= br
            connected[i] = avg_dist
    #quit()
    #connected = np.array(connected)
    print (connected)

    if connected[max(connected, key=connected.get)] > 0.4: 
        return (False, connected)
    return (True, connected)


def fix_connected(connected, clusters_of_clusters):
    for i,c in enumerate(clusters_of_clusters):
        if i in connected.keys():
            if connected[i] > 0.4:
                clusters_of_clusters[i] = max(clusters_of_clusters) + 1
    return clusters_of_clusters


def nearest_neighbours(EMB, classes, image_names):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import NearestNeighbors

    EMB_correct = np.array([x for i, x in enumerate(EMB) if classes[i] >= 0])
    classes_correct = np.array([classes[i] for i, x in enumerate(EMB) if classes[i] >= 0])
    image_names_correct = np.array([image_names[i] for i, x in enumerate(EMB) if classes[i] >= 0])

    EMB_noise = np.array([x for i, x in enumerate(EMB) if classes[i] == -1])
    image_names_noise = np.array([image_names[i] for i, x in enumerate(EMB) if classes[i] == -1])

    n_neighbors = int(np.sqrt(len(classes)/(max(classes)+1)))
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto').fit(EMB_correct, classes_correct)

    classes_noise = neigh.predict(EMB_noise)

    # nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(EMB_correct)
    # br = 0
    # distances, indices = nbrs.kneighbors(EMB_noise)
    # for i in range(len(distances)):
    #     distance, indice = distances[i], indices[i]
    #     print (image_names_noise[br])
    #     br += 1
    #     for j, ind in enumerate(indice):
    #         print(image_names_correct[ind], classes_correct[ind], distance[j])

    return classes_noise, image_names_noise


def nearest_neighbours2(EMB, classes, clusters_mean, clusters_classes, image_names):
    from sklearn.neighbors import KNeighborsClassifier

    EMB_noise = np.array([x for i, x in enumerate(EMB) if classes[i] == -1])
    image_names_noise = np.array([image_names[i] for i, x in enumerate(EMB) if classes[i] == -1])

    neigh = KNeighborsClassifier(n_neighbors=1).fit(clusters_mean, clusters_classes)

    classes_noise = neigh.predict(EMB_noise)

    return classes_noise, image_names_noise


def analyze_embeddings(EMB, image_names = ''):    

    from sklearn.manifold import TSNE
    from sklearn.cluster import DBSCAN
    import hdbscan

    fnames = image_names

    try: 
        fnames = [x.decode('UTF-8') for x in fnames]
    except: pass

    start_time = time.time()

         
    print ('HDBSCAN fit...')
    min_cluster_size = int(len(EMB) / 100)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples = 2).fit(EMB) # za soft_clustering: , prediction_data=True
    clusters = clusterer.labels_
    print ('HDBSCAN done')

    # make_folders(clusters, datasetFolder, 'ClustersHDBSCAN', fnames)

    mat2D=EMB
    if precision_boost:
        print ('TSNE fit...')
        model = TSNE(init='pca', n_components=3, random_state=0, n_iter=300, perplexity=15, learning_rate=150, metric='cosine', method='exact', n_iter_without_progress=1000)    
        mat2D = model.fit_transform(EMB) 
        print ('TSNE done')


    clusters_mean = np.zeros((max(clusters)+1, len(EMB[0])))
    clusters_examples = np.zeros((max(clusters)+1, 1))
    for i, c in enumerate(clusters):
        if c < 0: 
            continue
        clusters_mean[c] += EMB[i]
        clusters_examples[c] += 1
    for i in range(len(clusters_mean)):
        clusters_mean[i] /= clusters_examples[i]

    print ('len of clusters_mean:', len(clusters_mean))


    number_of_clusters = max(clusters) + 1
    eps = 0.1
    possible_connections = []
    start = False
    while(1):
        clusters_of_clusters = DBSCAN(eps=eps, algorithm='brute', min_samples=1, metric='cosine').fit_predict(clusters_mean)
        print(clusters_of_clusters)
        # eps += 0.1
        # if not connections_valid(clusters_of_clusters, clusters_mean): 
        #     break
        #quit()
        if max(clusters_of_clusters) + 1 == number_of_clusters and not start:
            eps += 0.1
            possible_connections.append(clusters_of_clusters)
            continue
        elif max(clusters_of_clusters) + 1 == number_of_clusters or max(clusters_of_clusters) + 1 < number_of_clusters * 0.65 or not connections_valid(clusters_of_clusters, clusters_mean)[0]: 
            break
        else:
            start = True
            number_of_clusters = max(clusters_of_clusters) + 1
            possible_connections.append(clusters_of_clusters)
            eps += 0.1

    cv = connections_valid(clusters_of_clusters, clusters_mean)
    if not cv[0] and not max(clusters_of_clusters) + 1 < number_of_clusters * 0.65:
        final_clusters = fix_connected(cv[1], clusters_of_clusters)
    else:
        final_clusters = possible_connections[-1]
    print ('FINAL: ', final_clusters)
    clusters = [final_clusters[c] if c >= 0 else c for c in clusters]
   
    #classes_noise, image_names_noise = nearest_neighbours(mat2D, clusters, image_names)
    classes_noise, image_names_noise = nearest_neighbours2(mat2D, clusters, clusters_mean, final_clusters, image_names)
    br = 0
    for i, c in enumerate(clusters):
        if c < 0:
            clusters[i] = classes_noise[br]
            br += 1
    clusters = [final_clusters[c] for c in clusters]
 
    return clusters

    print ('Time of execution: ', time.time() - start_time)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Analyses embeddings from npy (embeddings represents images).")
        
    parser.add_argument("embeddings_path", type=str, help="path to embeddings.npy (image_names.npy must be next to it)", nargs='?')
    parser.add_argument("-imgf", type=str, help="if set, generates folders with clusters (copies images)", nargs=1)

    parser.add_argument("-n", help="number of images to analyse", type=int, nargs=1)

    parser.add_argument("--p", action="store_true", help="precision_boost - improves precision of noise clustering, increases execution time")

    args = parser.parse_args()  
    
    if not args.imgf:
        dataset_folder = ''
    else:
        dataset_folder = args.imgf[0]

    if args.embeddings_path is None:
        # LOG_DIR_name = os.path.split(cfg.DATASET_FOLDER)
        df = dataset_folder
        if df == '': df = cfg.DATASET_FOLDER
        LOG_DIR_name = './embeddings_' + os.path.split(df.strip('/').strip('\\').strip('\\\\').strip('//'))[-1] + '.npy'
        embeddings_path = LOG_DIR_name
        image_names_path = embeddings_path.replace('embeddings_', 'image_names_')
    else:
        embeddings_path = args.embeddings_path
        print (embeddings_path)
        LOG_DIR_name = os.path.split(embeddings_path)
        image_names_path = embeddings_path.replace('embeddings_', 'image_names_')

    if args.p:  
        print("Method: precision_boost")
        precision_boost = True
    if args.n:  
        print("Analyzing {0} images".format(str(args.n[0])))
        number_of_images = args.n[0]
        EMB = np.load(embeddings_path)[:number_of_images]
        image_names = np.load(image_names_path)[:number_of_images]
    else:
        EMB = np.load(embeddings_path)[:]
        image_names = np.load(image_names_path)[:]
        print("Analyzing all {0} images".format(str(len(EMB))))


    #dodati da ostavlja nesigurni sum vani

    print ('Embeddings and image names loaded.')

    clusters = analyze_embeddings(EMB, image_names) # za sve slike 
    if dataset_folder == '':
        LOG_DIR_name = os.path.split(embeddings_path)
        datasetFolder = os.path.join(LOG_DIR_name[0], LOG_DIR_name[1].strip('.npy') + '_clusters')
        outFile = open(datasetFolder+'.txt','w')
        for imgi in range(len(image_names)):
            outFile.write('\n')
            outFile.write(image_names[imgi] + '\n')
            outFile.write('Cluster: ' + str(clusters[imgi]) + '\n')
        outFile.close()
        print('Image labels: ', clusters)
        print('Clusters saved into: {0}'.format(datasetFolder))
    else:
        make_folders(clusters, dataset_folder, '_clusters', image_names)
