import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2 as cv

def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    #find warp of the gradients of It1
    im_grad_x = cv.Sobel(It1,cv.CV_64F,1,0)
    im_grad_y = cv.Sobel(It1,cv.CV_64F,0,1)

    X = np.arange(It.shape[0])
    Y = np.arange(It.shape[1])

    xv, yv = np.meshgrid(X,Y,indexing='ij')
    xv = np.hstack(xv)
    yv = np.hstack(yv)
    p_delta = np.ones(6)

    while np.linalg.norm(p_delta)>0.1:

        It1_warped = affine_transform(It1,M)

        mask = np.ones(It.shape, dtype=bool)
        mask_warped = affine_transform(mask,M)

        #error to be minimized
        error = (mask_warped*It) - It1_warped

        It1_grad_x_warped = affine_transform(im_grad_x,M)
        It1_grad_y_warped = affine_transform(im_grad_y,M)

        im_warped_grad_x = It1_grad_x_warped.reshape(-1,)
        im_warped_grad_y = It1_grad_y_warped.reshape(-1,)

        im_gradient = np.column_stack((im_warped_grad_x,im_warped_grad_y))

        steepest_descent = np.array((yv*im_gradient[:,0],yv*im_gradient[:,1],xv*im_gradient[:,0],xv*im_gradient[:,1],im_gradient[:,0],im_gradient[:,1]))
        steepest_descent = steepest_descent.T        #shape 76800x6
        hessian = np.dot(steepest_descent.T,steepest_descent)
        error = error.reshape(-1,)

        value = np.dot(steepest_descent.T,error)

        p_delta = np.dot(np.linalg.inv(hessian),value)

        M[0][0]+=p_delta[0]
        M[0][1]+=p_delta[2]
        M[0][2]+=p_delta[4]
        M[1][0]+=p_delta[1]
        M[1][1]+=p_delta[3]
        M[1][2]+=p_delta[5]

    return M
