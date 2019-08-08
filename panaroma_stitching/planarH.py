import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def stackMatrix(p1,p2):
    A_matrix = []
    x1 = p1[0,0]
    y1 = p1[1,0]
    u1 = p2[0,0]
    v1 = p2[1,0]

    x2 = p1[0,1]
    y2 = p1[1,1]
    u2 = p2[0,1]
    v2 = p2[1,1]

    x3 = p1[0,2]
    y3 = p1[1,2]
    u3 = p2[0,2]
    v3 = p2[1,2]

    x4 = p1[0,3]
    y4 = p1[1,3]
    u4 = p2[0,3]
    v4 = p2[1,3]

    a = [-u1,-v1,-1,0,0,0,x1*u1,x1*v1,x1]
    A_matrix.append(a)
    b = [0,0,0,-u1,-v1,-1,u1*y1,v1*y1,y1]
    A_matrix.append(b)
    a = [-u2,-v2,-1,0,0,0,x2*u2,x2*v2,x2]
    A_matrix.append(a)
    b = [0,0,0,-u2,-v2,-1,u2*y2,v2*y2,y2]
    A_matrix.append(b)
    a = [-u3,-v3,-1,0,0,0,x3*u3,x3*v3,x3]
    A_matrix.append(a)
    b = [0,0,0,-u3,-v3,-1,u3*y3,v3*y3,y3]
    A_matrix.append(b)
    a = [-u4,-v4,-1,0,0,0,x4*u4,x4*v4,x4]
    A_matrix.append(a)
    b = [0,0,0,-u4,-v4,-1,u4*y4,v4*y4,y4]
    A_matrix.append(b)

    return A_matrix

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    A_matrix = stackMatrix(p1,p2)
    A = np.vstack(A_matrix)
    u,s,v = np.linalg.svd(A)
    H2to1 = v[-1].reshape(3,3)
    H2to1 = H2to1/H2to1[2,2]

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    ###########################
    # TO DO ...
    bestH = np.zeros((3,3))
    max_inliers = 0

    for i in range(num_iter):
        p1 = []
        p2 = []
        outlier_dist = 0
        rand_points = np.random.randint(0,matches.shape[0],4)
        for x in rand_points:
            p1.append(locs1[matches[x,0],0:2])
            p2.append(locs2[matches[x,1],0:2])
        p1 = np.transpose(np.vstack(p1))
        p2 = np.transpose(np.vstack(p2))
        H = computeH(p1,p2)

        pts2 = locs2[matches[:,1],0:2]
        row_size = pts2.shape[0]
        row_matrix = np.ones(row_size).reshape(-1,1)
        feature2 = np.hstack((pts2,row_matrix))
        feature2 = np.transpose(feature2)
        predicted = np.matmul(H,feature2)
        predicted = np.true_divide(predicted, predicted[-1])
        actual = locs1[matches[:,0],0:2]
        distance = np.subtract(actual,np.transpose(predicted[0:2]))
        distance = np.linalg.norm(distance,axis=1)
        num_inliers = distance[distance<tol].shape[0]

        if(num_inliers>max_inliers):
            max_inliers = num_inliers
            bestH = H

    return bestH



if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
