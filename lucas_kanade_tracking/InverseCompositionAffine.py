import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2 as cv


def InverseCompositionAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    temp_grad_x = cv.Sobel(It,cv.CV_64F,1,0)
    temp_grad_y = cv.Sobel(It,cv.CV_64F,0,1)

    temp_grad_x = temp_grad_x.reshape(-1,)
    temp_grad_y = temp_grad_y.reshape(-1,)

    temp_gradient = np.column_stack((temp_grad_x,temp_grad_y))

    X = np.arange(It.shape[0])
    Y = np.arange(It.shape[1])

    xv, yv = np.meshgrid(X,Y,indexing='ij')
    xv = np.hstack(xv)
    yv = np.hstack(yv)

    steepest_descent = np.array((yv*temp_gradient[:,0],yv*temp_gradient[:,1],xv*temp_gradient[:,0],xv*temp_gradient[:,1],temp_gradient[:,0],temp_gradient[:,1]))
    steepest_descent = steepest_descent.T

    hessian = np.dot(steepest_descent.T,steepest_descent)

    p_delta = np.ones(6)

    while np.linalg.norm(p_delta)>0.1:

        im_warped = affine_transform(It1,M)

        mask = np.ones(It.shape, dtype=bool)
        mask_warped = affine_transform(mask,M)

        #error to be minimized
        error = (mask_warped*It) - im_warped

        error = error.reshape(-1,)

        value = np.dot(steepest_descent.T,error)

        p_delta = np.dot(np.linalg.inv(hessian),value)

        M_delta = np.array(([1+p_delta[0],p_delta[2],p_delta[4]],[p_delta[1],1+p_delta[3],p_delta[5]],[0.0,0.0,1.0]))

        M = np.dot(M,np.linalg.inv(M_delta))

    return M
