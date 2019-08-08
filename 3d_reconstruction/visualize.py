'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper as hp
from mpl_toolkits.mplot3d import Axes3D


def main():

    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    K_matrices = np.load('../data/intrinsics.npz')
    data = np.load('../data/templeCoords.npz')

    pts1 = data['x1']
    pts1 = np.concatenate([pts1,data['y1']],axis=1)
    M = 640
    F = np.array([[ 9.78833285e-10, -1.32135929e-07,  1.12585666e-03],
       [-5.73843315e-08,  2.96800276e-09, -1.17611996e-05],
       [-1.08269003e-03,  3.04846703e-05, -4.47032655e-03]])

    K1 = K_matrices['K1']
    K2 = K_matrices['K2']
    pts2 = []
    for pt in pts1:
        x,y = sub.epipolarCorrespondence(im1, im2, F, pt[0], pt[1])
        pts2.append([x,y])
    pts2 = np.vstack(pts2)

    M1 = np.concatenate([np.eye(3), np.zeros([3, 1])], axis=1)
    C1 = np.dot(K1,M1)
    M2 = np.array([[ 0.99942701,  0.03331428,  0.0059843 , -0.02601138],
       [-0.03372743,  0.96531375,  0.25890503, -1.        ],
       [ 0.00284851, -0.25895852,  0.96588424,  0.07981688]])
    C2 = np.dot(K2,M2)
    '''
    C1 = np.array([[1.5204e+03, 0.0000e+00, 3.0232e+02, 0.0000e+00],
       [0.0000e+00, 1.5259e+03, 2.4687e+02, 0.0000e+00],
       [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00]])

    C2 = np.array([[ 1.52038999e+03, -2.76373041e+01,  3.01104652e+02,  -1.54174686e+01],
                   [-5.07614677e+01,  1.40904317e+03,  6.33511035e+02,  -1.50619561e+03],
                   [ 2.84850950e-03, -2.58958519e-01,  9.65884243e-01,   7.98168772e-02]])
    '''
    pts_3D, error = sub.triangulate(C1, pts1, C2, pts2)
    plt_pt = pts_3D[pts_3D[:,2]>3.2]

    np.savez('../results/q4_2.npz',F=F,M1=M1,M2=M2,C1=C1,C2=C2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plt_pt[:,0], plt_pt[:,1], plt_pt[:,2], c='r', marker='o')
    plt.show()


if __name__=="__main__":
    main()
