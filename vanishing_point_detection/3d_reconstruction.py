import scipy.io
import glob
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

from detect_vp import cluster_lines, find_VP, show_clusters
from camera_properties import eightpoint, sevenpoint, essentialMatrix, triangulate, epipolarCorrespondence
import helper as hp

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def stack_up_intrinsics(point1, point2):
    
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    return np.array([(x1*x2+y1*y2), (z1*x2+x1*z2), (z1*y2+y1*z2), (z1*z2)])

def intrinsics_calibration(points):

    '''
    Input: Vanishing points for an image

    Output: Camera intrinsic matrix 

    Description: Using the three orthogonal vanishing points in an image
                 camera matrix is calculated
    '''

    row_1 = stack_up_intrinsics(points[0], points[1])
    row_2 = stack_up_intrinsics(points[1], points[2])
    row_3 = stack_up_intrinsics(points[2], points[0])

    A = np.array([row_1, row_2, row_3])

    u, s, v = np.linalg.svd(A)

    w_11, w_31, w_32, w_33 = v[-1]

    W = np.array([(w_11, 0, w_31),
                  (0, w_11, w_32),
                  (w_31, w_32, w_33)])

    if not is_pos_def(W):
        sys.exit('Matrix for Cholesky decomposition is not positive definite. Camera intrinsics cannot be determined.')

    # Need to find what is the right form of K. 
    K_inv = np.linalg.cholesky(W).T
    K = np.linalg.inv(K_inv)
    K = K/K[-1][-1]

    return K

# This function is adapted from OpenCV tutorials
def feature_matching(img1, img2, show_matches=False):
    
    sift = cv2.xfeatures2d.SIFT_create(400)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)   # Distances: cv2.NORM_HAMMING
    bf = cv2.BFMatcher()   # Distances: cv2.NORM_HAMMING
    matches = bf.knnMatch(des1,des2,k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    pts1 = []
    pts2 = []

    for match in good_matches:
        pts1.append(kp1[match.queryIdx].pt) 
        pts2.append(kp2[match.trainIdx].pt)
    
    if show_matches:
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        plt.show()
    
    pts1 = np.vstack(pts1)
    pts2 = np.vstack(pts2)

    return pts1, pts2

def find_F(pts1, pts2, M):

    F = eightpoint(pts1, pts2, M)

    return F

def find_E(F, K1, K2):

    E = essentialMatrix(F, K1, K2)
    
    return E

def find_best_C2(C2s):

    C_idx = None
    min_error = sys.maxsize

    for idx in range(C2s.shape[-1]):
        M2 = np.dot(K2,C2s[:,:,idx])
        pts_3D, error = triangulate(M1, pts1, M2, pts2)

        if error < min_error:
            C_idx = idx

    C2 = C2s[:,:,C_idx]

    return C2

def plot_3D_points(pts_3D):

    # plt_pt = pts_3D[pts_3D[:,2]>0]
    plt_pt = pts_3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plt_pt[:,0], plt_pt[:,1], plt_pt[:,2], c='r', marker='o')
    plt.show()

def extrinsic_calibration(img1, img2, K1, K2):
    '''
    Input: 1. img1, img2 - Two images
           2. K1, K2 - Camera intrinsics

    Output: M1, M2 - Camera extrinsic matrix

    Description: Computes the camera extrinsics using two images and their intrinsics
                 by obtaining matching points and triangulation 
    '''

    pts1, pts2 = feature_matching(img1, img2)

    M = max(img1.shape[0], img1.shape[1])
    F = find_F(pts1, pts2, M)
    hp.displayEpipolarF(img1,img2,F)

    E = find_E(F, K1, K2)

    C1 = np.concatenate([np.eye(3), np.zeros([3, 1])], axis=1)
    M1 = np.dot(K1,C1)
    C2s = hp.camera2(E)

    C2 = find_best_C2(C2s)

    M2 = np.dot(K2,C2)
    pts_3D, error = triangulate(M1, pts1, M2, pts2)

    plot_3D_points(pts_3D)

    return M1, M2

def stack_up_homography(x, y, u, v):

    x = np.array([[-u, -v, -1, 0, 0, 0, x*u, x*v, x],
                 [0, 0, 0, -u, -v, -1, u*y, v*y, y]])
    
    return x

def find_A_homography(p1, p2):

    x1, x2, x3, x4, y1, y2, y3, y4 = np.extract(np.ones(p1.shape),p1)
    u1, u2, u3, u4, v1, v2, v3, v4 = np.extract(np.ones(p2.shape),p2)
    
    A = stack_up_homography(x1, y1, u1, v1)
    A = np.concatenate((A, stack_up_homography(x2, y2, u2, v2)), axis=0)
    A = np.concatenate((A, stack_up_homography(x3, y3, u3, v3)), axis=0)
    A = np.concatenate((A, stack_up_homography(x4, y4, u4, v4)), axis=0)

    return A

def compute_H(p1, p2):

    '''
    Inputs: p1, p2 - Each are size (2xN) matrices of corresponding (x, y)'
                     coordinates between two images

    Output: H2to1 - Homography matrix (3x3) 

    Description: Computes homography matrix that best matches the linear
            equation
    '''

    A = find_A_homography(p1, p2)
    u,s,v = np.linalg.svd(A)
    H2to1 = v[-1].reshape(3,3)
    H2to1 = H2to1/H2to1[2,2]

    return H2to1

def plane_RANSAC(pts1, pts2, iterations=50, tolerance=1):
    '''
    Inputs: 1. pts1, pts2 - two sets of corrsponding matches in each image
            2. iterations - number of iterations to run RANSAC
            3. tolerance - tolerance value for considering a point to be an inlier
    
    Output: Homography matrix with the most inliers found during RANSAC
    
    Description: Computes the best homography by computing the best set of matches using RANSAC
    '''
    bestH = None
    max_inliers = 0

    print("Extracting planes from images...\n")

    for i in range(iterations):

        random_idx = np.random.randint(low=0, high=len(pts1), size=4)
        
        p1 = pts1[random_idx].T
        p2 = pts2[random_idx].T

        H = compute_H(p1, p2)

        pts2_homogenized = np.concatenate((pts2, np.ones(len(pts2)).reshape(-1,1)), axis=1)
        pts1_estimated = np.matmul(H, pts2_homogenized.T)

        pts1_estimated = (pts1_estimated/pts1_estimated[-1]).T

        error = np.linalg.norm(pts1 - pts1_estimated[:,0:2], axis=1)

        num_inliers = len(error[error < tolerance])

        if(num_inliers > max_inliers):
            pts1_inliers = np.squeeze(pts1[np.argwhere(error < tolerance)])
            pts2_inliers = np.squeeze(pts2[np.argwhere(error < tolerance)])
            bestH = H
            max_inliers = num_inliers

    return bestH, pts1_inliers, pts2_inliers

def extract_planes(img1, img2):
    '''
    Input: Two images

    Output:

    Description: Extracts the homography of the common planes between two images
                 using RANSAC
    '''
    
    pts1, pts2 = feature_matching(img1, img2)

    H_matrices = []
    pts1_inliers_list = []
    pts2_inliers_list = []

    all_planes_extracted = False
    plane_threshold = 4           # Number of inlier points to be considered as a plane

    while not all_planes_extracted:
        
        H, pts1_inliers, pts2_inliers = plane_RANSAC(pts1, pts2)

        if len(pts1_inliers) < plane_threshold:
            all_planes_extracted = True
        else:
            H_matrices.append(H)
            pts1_inliers_list.append(pts1_inliers)
            pts2_inliers_list.append(pts2_inliers)
            ####I need to remove the inliers from pts1 and pts2

    num_planes = len(H)

    print("{} planes extracted\n".format(num_planes))

    import pdb;pdb.set_trace()


def main():

    img_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/input'
    lines_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/lines'

    images = glob.glob(img_folder+'/*')

    image1 = images[-5]
    image2 = images[-1]
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    img_name1 = (image1.split('/')[-1]).split('.')[0]
    img_name2 = (image2.split('/')[-1]).split('.')[0]

    # lines are of the structure (x1,x2,y1,y2,theta,r)
    lines1 = scipy.io.loadmat(lines_folder+'/'+img_name1+'.mat')['lines']
    lines2 = scipy.io.loadmat(lines_folder+'/'+img_name2+'.mat')['lines']

    # histogram_values gives the number of sorted_lines in each bin.  
    # clusters1 = cluster_lines(lines1)
    # clusters2 = cluster_lines(lines2)

    # show_clusters(img, clusters)
    # new_clusters1, VP1 = find_VP(clusters1)
    # new_clusters2, VP2 = find_VP(clusters2)
    # show_clusters(img1, new_clusters1, points=VP1, show_point=True)
    # show_clusters(img2, new_clusters2, points=VP2, show_point=True)

    # assert len(VP1) == 3, 'More than 3 cluster of lines formed.'
    # assert len(VP2) == 3, 'More than 3 cluster of lines formed.'

    # K1 = intrinsics_calibration(VP1)
    # K2 = intrinsics_calibration(VP2)

    # M1, M2 = extrinsic_calibration(img1, img2, K1, K2)    

    extract_planes(img1, img2)

    

if __name__ == '__main__':
    main()

