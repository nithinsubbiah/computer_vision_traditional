import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    #pano_im = np.zeros(im1.shape[0],im1.shape[1]).uint8
    im2_warp = cv2.warpPerspective(im2,H2to1,(im1.shape[0]+im2.shape[0],im1.shape[1]+im2.shape[1]))
    im1_warp = cv2.warpPerspective(im1,np.eye(3),(im1.shape[0]+im2.shape[0],im1.shape[1]+im2.shape[1]))
    pano_im = np.maximum(im1_warp,im2_warp)

    return pano_im

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix without cliping.
    '''
    ######################################
    # TO DO ...
    r1,c1,_ = im1.shape
    r2,c2,_ = im2.shape

    A = np.array([[0,0,1],[0,r2,1],[c2,0,1],[c2,r2,1]])
    pts = np.matmul(H2to1,np.transpose(A))
    pts = np.true_divide(pts[0:2],pts[2])
    canvas = np.max(pts,axis=1)
    width = int(canvas[0])
    height = int(canvas[1]-np.min(pts,axis=1)[1])

    width_des = width
    ratio = width/height
    height_des = int(width_des/ratio)
    scale = height_des/height

    translation_matrix = np.array([[1,0,0],[0,1,-np.min(pts,axis=1)[1]],[0,0,1/scale]]).astype('float')


    im2_warp = cv2.warpPerspective(im2,np.matmul(translation_matrix,H2to1),(width_des,height_des))
    im1_warp = cv2.warpPerspective(im1,translation_matrix,(width_des,height_des))
    pano_im = np.maximum(im1_warp,im2_warp)

    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    return pano_im



if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    #print(im1.shape)
    #locs1, desc1 = briefLite(im1)
    #locs2, desc2 = briefLite(im2)
    #matches = briefMatch(desc1, desc2)
    #plotMatches(im1,im2,matches,locs1,locs2)
    #H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    #pano_im = imageStitching_noClip(im1, im2, H2to1)
    #pano_im = imageStitching(im1, im2, H2to1)
    #np.save('../results/q6_1.npy', H2to1)

    im3 = generatePanorama(im1, im2)

    cv2.imwrite('../results/q6_3.jpg', im3)
    cv2.imshow('panoramas', im3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
