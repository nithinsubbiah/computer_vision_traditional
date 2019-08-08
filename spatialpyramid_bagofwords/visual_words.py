import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
from sklearn.cluster import KMeans
import scipy.spatial.distance
import os,time
import util
import random
import skimage.io
from tempfile import TemporaryFile

def check_normalization(image):
    if np.any(image) > 1:
        return True

    return False




def extract_filter_responses(image):

    normalize = check_normalization(image)

    if normalize == True:
        image = image.astype('float')/255

    if len(image.shape) == 2:
        image = np.tile(image[:, newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)

    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):
    size = image.shape
    imgs = extract_filter_responses(image)
    imgs = imgs.reshape(-1,60)

                #import pdb; pdb.set_trace();
    dist = scipy.spatial.distance.cdist(imgs,dictionary,metric='euclidean')
    pixel_values = np.argmin(dist,axis=1).reshape(size[0],size[1])


    return pixel_values



    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''




def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    idx,alpha,path = args
    im = skimage.io.imread("../data/"+path)
    im = im.astype('float')/255

    filter_response = extract_filter_responses(im)
    filter_response = filter_response.reshape(-1,60)
    l = filter_response.shape
    x = np.random.choice(l[0],alpha,replace='False')
    filter_response = filter_response[x,:]

    dictionary = TemporaryFile()
    np.save('../dict/'+str(idx)+'.npy',filter_response)





def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    train_data = np.load("../data/train_data.npz")
    #import pdb;pdb.set_trace()
    alpha = 100
    pool = multiprocessing.Pool(num_workers)
    #arg = [[idx, alpha, path] for idx,path in enumerate(train_data['files'])]
    #pool.map(compute_dictionary_one_image,arg)
    x = len(train_data['files'])
    ar = np.empty([alpha*x,60])
    for i in range(x):
        ar[i*alpha:(i+1)*alpha,:] = np.load("../dict/"+str(i)+".npy")
    #kmeans = sklearn.cluster.KMeans(n_clusters=200).fit(ar)
    #dictionary = kmeans.cluster_centers_
    #np.save('dictionary.npy',dictionary)
    dict = np.load("dictionary.npy")
