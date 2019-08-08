import numpy as np
import LucasKanadeAffine
from scipy.ndimage import affine_transform
import cv2
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import InverseCompositionAffine


def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1
	# Output:
	#	mask: [nxm]
    # put your implementation here

    mask = np.ones(image1.shape, dtype=bool)

    M = LucasKanadeAffine.LucasKanadeAffine(image1,image2)
    #M = InverseCompositionAffine.InverseCompositionAffine(image1,image2)

    image2_warped = affine_transform(image2,M)

    mask_2 = np.ones(image1.shape, dtype=bool)
    mask_2_warped = affine_transform(mask_2,M)

    error = image1 - image2_warped
    error = np.multiply(error,mask_2_warped)

    threshold = 0.3

    mask = error>threshold

    #mask = binary_erosion(mask,structure=np.ones((3,3)),iterations=2)
    #mask = binary_dilation(mask,structure=np.ones((2,2)),iterations=1)

    return mask
