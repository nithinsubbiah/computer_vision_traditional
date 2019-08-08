'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper as hp

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
K_matrices = np.load('../data/intrinsics.npz')

N = data['pts1'].shape[0]
M = 640

K1 = K_matrices['K1']
K2 = K_matrices['K2']

F = sub.eightpoint(data['pts1'], data['pts2'], M)
E = sub.essentialMatrix(F, K1, K2)
M1 = np.concatenate([np.eye(3), np.zeros([3, 1])], axis=1)
C1 = np.dot(K1,M1)
M2s = hp.camera2(E)
'''
for i in range(M2s.shape[-1]):
    C2 = np.dot(K2,M2s[:,:,i])
    pts_3D, error = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
    print(error)
'''
M2 = M2s[:,:,2]
C2 = np.dot(K2,M2)
pts_3D, error = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
#np.savez('../results/q3_3.npz',M2=M2,C2=C2,P=pts_3D)

hp.epipolarMatchGUI(im1, im2, F)
