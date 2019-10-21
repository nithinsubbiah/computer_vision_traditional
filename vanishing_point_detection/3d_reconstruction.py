import scipy.io
import glob
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import sys

from detect_vp import cluster_lines, find_VP, show_clusters

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def stack_up(point1, point2):
    
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    return np.array([(x1*x2+y1*y2), (z1*x2+x1*z2), (z1*y2+y1*z2), (z1*z2)])

def intrinsics_calibration(points):

    print(points)

    row_1 = stack_up(points[0], points[1])
    row_2 = stack_up(points[1], points[2])
    row_3 = stack_up(points[2], points[0])

    A = np.array([row_1, row_2, row_3])

    u, s, v = np.linalg.svd(A)

    w_11, w_31, w_32, w_33 = v[-1]

    W = np.array([(w_11, 0, w_31),
                  (0, w_11, w_32),
                  (w_31, w_32, w_33)])

    if not is_pos_def(W):
        sys.exit('Matrix for Cholesky decomposition is not positive definite. Camera intrinsics cannot be determined.')

    K = np.linalg.cholesky(W)

    return K

def main():

    img_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/input'
    lines_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/lines'

    images = glob.glob(img_folder+'/*')

    image = images[-5]
    img = cv2.imread(image)

    img_name = (image.split('/')[-1]).split('.')[0]

    # lines are of the structure (x1,x2,y1,y2,theta,r)
    lines = scipy.io.loadmat(lines_folder+'/'+img_name+'.mat')['lines']

    # histogram_values gives the number of sorted_lines in each bin.  
    clusters = cluster_lines(lines)

    # show_clusters(img, clusters)
    new_clusters, VP = find_VP(clusters)
    show_clusters(img, new_clusters, points=VP, show_point=True)

    assert len(VP) == 3, 'More than 3 cluster of lines formed.'

    K = intrinsics_calibration(VP)

    print(K)

if __name__ == '__main__':
    main()

