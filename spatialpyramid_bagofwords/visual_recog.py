import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io
import multiprocessing



def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''



    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    SPM_layer_num = 2
    K = 200
    args = [[path,dictionary,SPM_layer_num,K] for path in train_data['files']]
    labels = [l for l in train_data['labels']]
    pool = multiprocessing.Pool(num_workers)


    features = pool.map(get_image_feature,args)

    np.savez('trained_system.npz',dictionary=dictionary,features=features,labels=labels,SPM_layer_num=SPM_layer_num)




def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")

    #Get features from test data
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    train_labels = trained_system['labels']
    ########################################### HOW TO EXTRACT FROM MODEL???
    SPM_layer_num = 2
    K = 200
    args = [[path,dictionary,SPM_layer_num,K] for path in test_data['files']]
    labels = [l for l in test_data['labels']]
    pool = multiprocessing.Pool(num_workers)
    test_features = pool.map(get_image_feature,args)
    #import pdb;pdb.set_trace()
    correct = 0
    conf = np.zeros((8,8))

    #Get output for the test data
    for idx, feature in enumerate(test_features):
        a = distance_to_set(feature,trained_features)
        val = np.argmax(a)
        target = train_labels[val]
        output = labels[idx]
        #import pdb; pdb.set_trace();
        conf[target][output]+=1

        if output == target:
            correct+=1

    accuracy = correct/(idx+1)


    return conf, accuracy



def get_image_feature(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    file_path,dictionary,layer_num,K = args
    image = skimage.io.imread("../data/"+file_path)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image,dictionary)
    SPM_values = get_feature_from_wordmap_SPM(wordmap,layer_num,K)

    return SPM_values

    # ----- TODO -----


def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    sim = np.minimum(word_hist,histograms)
    sim = np.sum(sim,axis=1)

    return sim



    # ----- TODO -----



def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----

    hist, bin_edge = np.histogram(wordmap,bins=dict_size,range=(0,dict_size),density='True')

    return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----

    h,w=wordmap.shape

    for k in range(layer_num,-1,-1):
        for i in range(2**k):
            for j in range(2**k):
                smallest_map = wordmap[i*(h//2**k):(i+1)*(h//2**k),j*(w//2**k):(j+1)*(w//2**k)]
                if i+j == 0:
                    histogram_values = get_feature_from_wordmap(smallest_map,dict_size)
                else:
                    histogram_values = np.append(histogram_values,get_feature_from_wordmap(smallest_map,dict_size))
        histogram_values = np.true_divide(histogram_values, 4**k)
        norm_factor = 2**(k-layer_num-1)
        if k==0: norm_factor=0.25
        histogram_values = np.multiply(histogram_values, norm_factor)

        if k==layer_num:
            hist_all = histogram_values
        else:
            hist_all = np.append(hist_all,histogram_values)

    return hist_all
