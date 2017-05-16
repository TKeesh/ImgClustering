from visualization_config import cfg

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc
import numpy as np

from glob import glob
from scipy import spatial
from sklearn import mixture
from sklearn.decomposition import PCA

import os, time

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
    LOG_DIR_name = os.path.split(LOG_DIR)
    np.save(os.path.join(LOG_DIR_name[0], 'image_names_' + LOG_DIR_name[1]), im_names)
    np.save(os.path.join(LOG_DIR_name[0], 'embeddings_' + LOG_DIR_name[1]), EMB)

    # projector run
    projector.visualize_embeddings(summary_writer, config)

    # save embeddings
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), 1)

    # write metadata
    if len(EMB[0]) == 1000:
        metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
        metadata_file.write('Name\tClass\n')
        cnf = open(os.path.join(LOG_DIR_name[0], 'classes.txt'), 'w')
        for i, name in enumerate(image_names):
            prob = EMB[i]
            pred = np.argsort(prob)[::-1]
            metadata_file.write('%06d\t%s\n' % (i, name+': '+' '.join(synset[pred[0]].split()[1:])))
            cnf.write(name + ': ')
            topX = [' '.join(synset[pred[i]].split()[1:]) for i in range(7)]
            #print (topX)
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
    datasetFolder = datasetFolder.strip('.').strip('/')
    outFile = open(datasetFolder+'.txt','w')
    for imgi in range(len(fnames)):
        outFile.write('\n')
        outFile.write(fnames[imgi] + '\n')
        outFile.write('Cluster: ' + str(clusters[imgi]) + '\n')
    outFile.close()
    if os.path.exists(folder): 
        import shutil
        shutil.rmtree(folder)
    for imgi in range(len(clusters)):
        # print (fnames[imgi])
        if not os.path.exists(folder + '\\' + str(clusters[imgi])):
            os.makedirs(folder + '\\' + str(clusters[imgi]))
        imgorg = glob(os.path.join(datasetFolder, fnames[imgi].split('.')[0]) + '.*')[0]
        try: img = (scipy.misc.imread(imgorg)[:,:,:3]).astype('float32')
        except: img = (scipy.misc.imread(imgorg)[:]).astype('float32')
        scipy.misc.imsave(os.path.join(folder + '\\' + str(clusters[imgi]) + '\\', imgorg.split('\\')[-1]), img)

def create_folders_2(EMB, image_names = '', images_folder = ''):
    from sklearn.neighbors import NearestNeighbors

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

    for n_components in range(7, 15):
    # for n_components in [12]:
        
        print('fitting...')
        dpgmm = mixture.BayesianGaussianMixture(n_components=12, covariance_type='full').fit(EMB)
        
        print('predicting...')
        pred = dpgmm.predict(EMB)

        print('sampling...')
        new_samples = np.array(dpgmm.sample(EMB.shape[0])[0]) # iz dobivenih gaussova sempliram nove podatke koje cu usporedivati sa nasim podacima

        # print('new_samples: ', str(new_samples))
        # print('list(new_samples): ', str(list(new_samples)))

        avg_dist_1 = 0 # prosjecna udaljenost nekog primjera od njegovih K najblizih susjeda u ORIGINALNOM datasetu
        avg_dist_2 = 0 # prosjecna udaljenost nekog primjera od njegovih K najblizih susjeda u GENERIRANOM datasetu
        print('finding nearest neighbors...')
        for our_sample in EMB:
            n_neighbors = 5

            neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(EMB)
            distances, indices = neighbors.kneighbors(our_sample.reshape(1, -1))
            dist_1 = np.sum(distances) / n_neighbors # prosjecna udaljenost naseg primjera od njegovih 5 najblizih susjeda iz ORIGINALNOG skupa

            neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(new_samples)
            distances, indices = neighbors.kneighbors(our_sample.reshape(1, -1))
            dist_2 = np.sum(distances) / n_neighbors # prosjecna udaljenost naseg primjera od njegovih 5 najblizih susjeda iz GENERIRANOG skupa

            avg_dist_1 += dist_1
            avg_dist_2 += dist_2

        avg_dist_1 /= EMB.shape[0]
        avg_dist_2 /= EMB.shape[0]

        difference = np.absolute(avg_dist_1 - avg_dist_2)

        # ovo nas najvise zanima jer ako su izracunati gaussovi stvarno tocni, onda bi semplirani podaci trebali biti slicni originalnim podacima
        # pa bi ova razlika trebala biti mala, dok bi u protivnom ta razlika trebala biti veca
        # nadamo se da ce za n_components = 12 ispasti da je ta razlika najmanja, ili mozda da tad padne za veliki iznos ili tak nes
        print('n_components: ' + str(n_components))
        print('difference of original and sampled datasets: ' + str(difference))

        # print('pred: ' + str(pred))

    print('done')

    # make_folders(pred, datasetFolder, 'Clusters_GMM_3', fnames)  

def create_folders(EMB, image_names = '', images_folder = ''):    
    # from sklearn.cluster import DBSCAN   
    # from sklearn.manifold import TSNE
    

    # model = TSNE(init='pca', n_components=3, random_state=0, n_iter=310, perplexity=15, learning_rate=10, metric='cosine', method='exact', n_iter_without_progress=1000)    
    # np.set_printoptions(suppress=True)
    # print("TSNEfit")    
    # mat2D = model.fit_transform(EMB) 
   
    #mat2D = tsne(X = EMB.astype('float64'), no_dims = 3, initial_dims = len(EMB[0]), perplexity = 5.0)
    
    #print(mat2D.shape)

    # maxx = np.max(mat2D[:, 0])
    # minx = np.min(mat2D[:, 0])

    # maxy = np.max(mat2D[:, 1])
    # miny = np.min(mat2D[:, 1])

    # maxz = np.max(mat2D[:, 2])
    # minz = np.min(mat2D[:, 2])

    # print("maxx = " + str(maxx))
    # print("minx = " + str(minx))
    # print("maxy = " + str(maxy))
    # print("miny = " + str(miny))
    # print("maxz = " + str(maxz))
    # print("minz = " + str(minz))
    # eps = np.maximum(np.maximum(maxx-minx, maxy-miny), maxz-minz) / 10.0
    # eps = ((maxx-minx) + (maxy-miny) + (maxz-minz)) / 3.0 * 490000 / len(EMB) / len(EMB) / len(EMB)
    # print ('eps = ', eps)

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(mat2D[:,0], mat2D[:,1], mat2D[:,2], c='r')
    # fig.savefig('./TSNEplot.png')
    
    #mat2D = mat2D.transpose() 
    # print (type(mat2D), mat2D.shape)

    # clusters = [len(EMB)+1]
    # eps = 0.05
    # while(max(clusters)+1 > 16): #< 8 or max(clusters)+1 > 14
    #     clusters = DBSCAN(eps=eps, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    #     #clusters = DBSCAN(eps=eps, algorithm='brute', min_samples=1, metric='cosine').fit_predict(EMB).flatten()        
    #     print ('clusters: ', max(clusters)+1)
    #     print ('eps: ', eps)
    #     eps += 0.05
    # eps -= 0.05
    


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
    # # print (fnames[0], type(fnames))
  
    # pca = PCA(n_components=3)
    # pca.fit(EMB)
    # print('PCA components: ' + str(pca.get_params())) 
    dict_to_save = {'score' : [],
                    'components' : [],
                    'best_pred' : [],
                    'pred' : [],
                    'seconds' : 0}
    i = 5
    end = 0
    last_score = float('-inf')
    last_pred = []
    start_time = time.time()
    while(end < 5):
        print('fitting...')
        dpgmm = mixture.BayesianGaussianMixture(n_components=i, covariance_type='full').fit(EMB)
        print('predicting...')
        new_pred = dpgmm.predict(EMB)
        # print('precisions: ' + str(dpgmm.precisions_))
        # print('len(pred): ' + str(pred))        
        new_score = dpgmm.score(EMB)
        dict_to_save['components'].append(i)
        dict_to_save['score'].append(new_score)   
        dict_to_save['pred'].append(new_pred)   
        print('score: ' + str(new_score))
        print('done')  
        #quit()   
        if new_score < last_score: 
            end += 1 
            dict_to_save['best_pred'] = last_pred
        else:
            last_score = new_score
            last_pred = new_pred            
        i += 1
    dict_to_save['seconds'] = time.time() - start_time
    np.save(str(len(EMB)), dict_to_save)

    #make_folders(last_pred, datasetFolder, 'Clusters_GMM_3', fnames)
    # make_folders(clusters, datasetFolder, 'Clusters'+str(eps), fnames)





    # clusters_mean = np.zeros((max(clusters)+1, len(EMB[0])))
    # clusters_examples = np.zeros((max(clusters)+1, 1))
    # for i, c in enumerate(clusters):
    #     clusters_mean[c] += EMB[i]
    #     clusters_examples += 1
    # for i in range(len(clusters_mean)):
    #     clusters_mean[i] /= clusters_examples[i]


    # result = 1 - spatial.distance.cosine(EMB[1], EMB[3])
    # print("kosinusna_1008: " + str(result))
    # result = np.linalg.norm(EMB[1]-EMB[3])
    # print("euklidna_1008: " + str(result))

    # result = 1 - spatial.distance.cosine(mat2D[1], mat2D[3])
    # print("kosinusna_3: " + str(result))
    # result = np.linalg.norm(mat2D[1]-mat2D[3])
    # print("euklidna_3: " + str(result))

    # model = TSNE(n_components=3, random_state=0, n_iter=720, perplexity=5, learning_rate=10, metric='cosine')
    # print (clusters_mean.shape)
    # mat2D = model.fit_transform(clusters_mean) 

    # clusters_of_clusters = DBSCAN(eps=1.5, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # # print(clusters_of_clusters)
    # clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    # make_folders(clusters_of_clusters, datasetFolder, '1_5', fnames)
    # clusters_of_clusters = DBSCAN(eps=2, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # # print(clusters_of_clusters)
    # clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    # make_folders(clusters_of_clusters, datasetFolder, '2', fnames)
    # clusters_of_clusters = DBSCAN(eps=2.5, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # # print(clusters_of_clusters)
    # clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    # make_folders(clusters_of_clusters, datasetFolder, '2_5', fnames)
    # clusters_of_clusters = DBSCAN(eps=3, algorithm='ball_tree', min_samples=1, metric='euclidean').fit_predict(mat2D)
    # # print(clusters_of_clusters)
    # clusters_of_clusters = [clusters_of_clusters[c] for c in clusters]
    # make_folders(clusters_of_clusters, datasetFolder, '3', fnames)


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
