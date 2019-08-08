import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift
import cv2 as cv

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p = p0
    p_delta = np.ones(2)

    rect_width = rect[2]-rect[0]
    rect_height = rect[3]-rect[1]

    width = np.arange(0,It1.shape[0])
    height = np.arange(0,It1.shape[1])

    im_spline = RectBivariateSpline(width,height,It1)

    #find warp of the gradients of It1
    im_grad_x = cv.Sobel(It1,cv.CV_64F,1,0)
    im_grad_y = cv.Sobel(It1,cv.CV_64F,0,1)
    im_spline_grad_x = RectBivariateSpline(width,height,im_grad_x)
    im_spline_grad_y = RectBivariateSpline(width,height,im_grad_y)


    while np.linalg.norm(p_delta)>0.1:

        jacobian = np.eye(2)                    #jacobian for translation is just an identity
        #template is obtained from It
        #template = It[rect[1]:rect[3]+1,rect[0]:rect[2]+1]
        template = It
        #warp is updated using p and It1 is warped
        '''
        warp = rect[:]     #check if rect is passed as reference

        warp = np.add(warp,np.array((p[0],p[1],p[0],p[1])))

        X = np.arange(warp[0],warp[2]+1)[0:87]
        Y = np.arange(warp[1],warp[3]+1)[0:36]

        xv, yv = np.meshgrid(X,Y)

        im_warped = im_spline.ev(yv,xv)
        '''
        It1_shifted = shift(It1,np.flip(-p))

        im_warped = It1_shifted[rect[1]:rect[3]+1,rect[0]:rect[2]+1]

        #error to be minimized
        error = template-im_warped

        '''

        im_warped_grad_x = im_spline_grad_x.ev(yv,xv)
        im_warped_grad_y = im_spline_grad_y.ev(yv,xv)

        im_warped_grad_x = im_warped_grad_x.reshape(-1,)
        im_warped_grad_y = im_warped_grad_y.reshape(-1,)
        '''
        It1_grad_x_shifted = shift(im_grad_x,np.flip(-p))
        It1_grad_y_shifted = shift(im_grad_y,np.flip(-p))

        im_warped_grad_x = It1_grad_x_shifted[rect[1]:rect[3]+1,rect[0]:rect[2]+1]
        im_warped_grad_y = It1_grad_y_shifted[rect[1]:rect[3]+1,rect[0]:rect[2]+1]

        im_warped_grad_x = im_warped_grad_x.reshape(-1,)
        im_warped_grad_y = im_warped_grad_y.reshape(-1,)

        im_gradient = np.column_stack((im_warped_grad_x,im_warped_grad_y))

        steepest_descent = np.dot(im_gradient,jacobian)

        hessian = np.dot(steepest_descent.T,steepest_descent)
        error = error.reshape(-1,)

        value = np.dot(steepest_descent.T,error)

        p_delta = np.dot(np.linalg.inv(hessian),value)
        p[0] = p[0] + p_delta[0]
        p[1] = p[1] + p_delta[1]
        import pdb; pdb.set_trace()

    return p
