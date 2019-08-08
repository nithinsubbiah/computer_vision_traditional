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

# 2.1
#F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
#hp.displayEpipolarF(im1,im2,F8)
#F = F8

#2.2
F7 = sub.sevenpoint(data['pts1'][21:28, :], data['pts2'][21:28, :], M)
hp.displayEpipolarF(im1,im2,F7)

E = sub.essentialMatrix(F, K_matrices['K1'], K_matrices['K2'])
