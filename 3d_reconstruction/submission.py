"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper as hp

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):

    T = np.array(([1/M,0,0],[0,1/M,0],[0,0,1]))

    pts1_n = [np.dot(T,np.insert(pt1,2,1)).T for pt1 in pts1]
    pts2_n = [np.dot(T,np.insert(pt2,2,1)).T for pt2 in pts2]

    U = []

    for i in range(pts1.shape[0]):
        a = [pts1_n[i][0]*pts2_n[i][0],pts1_n[i][1]*pts2_n[i][0],pts2_n[i][0],pts1_n[i][0]*pts2_n[i][1],pts1_n[i][1]*pts2_n[i][1],pts2_n[i][1],pts1_n[i][0],pts1_n[i][1],1]
        U.append(a)

    U = np.vstack(U)

    u, s, v = np.linalg.svd(U)

    F = v[-1].reshape(3,3)

    pts1_n = np.array(pts1_n)
    pts2_n = np.array(pts2_n)

    pts1_n = np.delete(pts1_n,-1,axis=1)
    pts2_n = np.delete(pts2_n,-1,axis=1)

    F = hp.refineF(F,np.array(pts1_n),np.array(pts2_n))
    F = np.dot(T.T,np.dot(F,T))

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):

    T = np.array(([1/M,0,0],[0,1/M,0],[0,0,1]))

    pts1_n = [np.dot(T,np.insert(pt1,2,1)).T for pt1 in pts1]
    pts2_n = [np.dot(T,np.insert(pt2,2,1)).T for pt2 in pts2]

    U = []

    for i in range(pts1.shape[0]):
        a = [pts1_n[i][0]*pts2_n[i][0],pts1_n[i][1]*pts2_n[i][0],pts2_n[i][0],pts1_n[i][0]*pts2_n[i][1],pts1_n[i][1]*pts2_n[i][1],pts2_n[i][1],pts1_n[i][0],pts1_n[i][1],1]
        U.append(a)

    U = np.vstack(U)

    u, s, v = np.linalg.svd(U)

    F1 = v[-1].reshape(3,3)
    F2 = v[-2].reshape(3,3)

    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)

    a0 = fun(0)
    a1 = 2*(fun(1)-fun(-1))/3-(fun(2)-fun(-2))/12
    a2 = 0.5*fun(1)+0.5*fun(-1)-fun(0)
    a3 = ((fun(2)-fun(-2))-(2*(fun(1)-fun(-1))))/12

    roots = np.roots([a3,a2,a1,a0])

    F = (1-roots[0]*F1) + (roots[0]*F2)

    pts1_n = np.array(pts1_n)
    pts2_n = np.array(pts2_n)

    pts1_n = np.delete(pts1_n,-1,axis=1)
    pts2_n = np.delete(pts2_n,-1,axis=1)

    F = hp.refineF(F,np.array(pts1_n),np.array(pts2_n))
    F = np.dot(T.T,np.dot(F,T))

    return F

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):

    E = np.dot(K1.T,np.dot(F,K2))

    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):

    pts_3D = []
    error = 0

    for pt1, pt2 in zip(pts1,pts2):

        x1 = pt1[0]
        y1 = pt1[1]
        x2 = pt2[0]
        y2 = pt2[1]
        
        A = np.array([C1[2]*x1-C1[0],C1[2]*y1-C1[1],C2[2]*x2-C2[0],C2[2]*y2-C2[1]])
        u, s, v = np.linalg.svd(A)
        pt_3D = v[-1].T
        pt_3D = pt_3D/pt_3D[-1]
        pts_3D.append(pt_3D)
        '''
        projection_1 = np.dot(C1,pt_3D)[:-1]
        projection_2 = np.dot(C2,pt_3D)[:-1]
        '''
        projection_1 = np.dot(C1,pt_3D)
        projection_2 = np.dot(C2,pt_3D)

        projection_1 = projection_1/projection_1[-1]
        projection_2 = projection_2/projection_2[-1]
        error += np.linalg.norm(pt1-projection_1[:-1]) + np.linalg.norm(pt2-projection_2[:-1])

    pts_3D = np.vstack(pts_3D)
    pts_3D = np.delete(pts_3D,-1,axis=1)

    return pts_3D, error

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):

    l = np.dot(F,[x1,y1,1])
    sy, sx, _ = im2.shape
    ye = sy-1
    ys = 0
    window = im1[x1-3:x1+3,y1-3:y1+3]
    min_intensity = float('inf')

    y2 = ys
    x2 = int(-(l[1] * ys + l[2])/l[0])

    for y in range(ys,ye):
        x = int(-(l[1] * y + l[2])/l[0])
        window_2 = im2[x-3:x+3,y-3:y+3]
        if window_2.shape != window.shape or np.linalg.norm([x1-x,y1-y])>10 :
            continue
        intensity_difference = window - window_2
        if np.linalg.norm(intensity_difference)<min_intensity:
            min_intensity = np.linalg.norm(intensity_difference)
            x2 = x
            y2 = y

    return x2, y2
