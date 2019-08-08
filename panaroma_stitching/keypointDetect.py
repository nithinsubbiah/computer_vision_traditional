import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1,
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    DoG_pyramid = []
    for i in DoG_levels:
        DoG_pyramid.append(np.subtract(gaussian_pyramid[:,:,i+1],gaussian_pyramid[:,:,i]))
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)

    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = []
    ##################
    # TO DO ...
    # Compute principal curvature here
    for i in range(DoG_pyramid.shape[-1]):
        im = np.array(DoG_pyramid[:,:,i])
        sobel_x = cv2.Sobel(im,-1,2,0,ksize=3)
        sobel_y = cv2.Sobel(im,-1,0,2,ksize=3)
        sobel_xy = cv2.Sobel(im,-1,1,0,ksize=3)
        sobel_xy = cv2.Sobel(sobel_xy,-1,0,1,ksize=3)
        sobel_yx = cv2.Sobel(im,-1,0,1,ksize=3)
        sobel_yx = cv2.Sobel(sobel_yx,-1,1,0,ksize=3)

        trace = np.add(sobel_x,sobel_y)
        trace = np.square(trace)
        det = np.subtract(np.multiply(sobel_x,sobel_y),np.multiply(sobel_yx,sobel_xy))
        det=det+1e10
        principal_curvature.append(np.true_divide(trace,det))

    principal_curvature = np.stack(principal_curvature,axis=-1)

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    check = 0


    ##############
    #  TO DO ...
    # Compute locsDoG here
    for i in DoG_levels:
        ratio = np.array(principal_curvature[:,:,i])
        im = np.array(DoG_pyramid[:,:,i])
        indices_thr = np.where(ratio<th_r)
        indices_thc = np.where(im>th_contrast)
        # indices = [(x,y) for (x,y) in zip(*indices_thc) if (x,y) in zip(*indices_thr)]
        indices_thr2 = {(x,y) for (x,y) in zip(*indices_thr)}
        indices_thc2 = {(x,y) for (x,y) in zip(*indices_thc)}
        indices = indices_thr2.intersection(indices_thc2)

        for j in indices:
            val_idx = im[j]
            x = j[0]
            y = j[1]
            if((np.amax(im[max(0,x-1):min(x+2,im.shape[0]),max(0,y-1):min(y+2,im.shape[1])])==val_idx or
                np.amin(im[max(0,x-1):min(x+2,im.shape[0]),max(0,y-1):min(y+2,im.shape[1])])==val_idx) and
               (np.amax(DoG_pyramid[x,y,max(0,i-1):min(i+2,5)])==val_idx or
                np.amin(DoG_pyramid[x,y,max(0,i-1):min(i+2,5)])==val_idx)):
                extremaLocation = np.array([x,y,DoG_levels[i]])
                locsDoG.append(extremaLocation)

    locsDoG = np.vstack(locsDoG)

    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    im_pyr = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    for locs in locsDoG:
        cv2.circle(im,(locs[1], locs[0]),0,(0,255,0))
    #cv2.imshow('image',im)
    #cv2.waitKey(0)

    return locsDoG, im_pyr







if __name__ == '__main__':
    # test gaussian pyramid
    #levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    #im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    # test DoG pyramid
    #DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    # test compute principal curvature
    #pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    #th_contrast = 0.03
    #th_r = 12
    #locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
