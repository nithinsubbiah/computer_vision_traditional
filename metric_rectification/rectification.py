import numpy as np 
import glob
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import sys

# Check for positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# Returns the homogrphy that applies affine rectification
def affine_rectification(points):
    line1_1 = np.cross(points[0][0], points[0][1])
    line1_2 = np.cross(points[0][2], points[0][3])
    
    p1_infinity = np.cross(line1_1, line1_2)
    p1_infinity = p1_infinity/p1_infinity[-1]

    line2_1 = np.cross(points[1][0], points[1][1])
    line2_2 = np.cross(points[1][2], points[1][3])

    p2_infinity = np.cross(line2_1, line2_2)
    p2_infinity = p2_infinity/p2_infinity[-1]

    vanishing_line = np.cross(p1_infinity, p2_infinity)
    vanishing_line = (np.sign(vanishing_line[-1])*vanishing_line)/np.linalg.norm(vanishing_line)
    Ha = np.array([[1,0,0], [0,1,0], [vanishing_line[0],vanishing_line[1],vanishing_line[2]]])
    return Ha 

def metric_rectification(Ha, points):

    # Get two sets of perpendicular lines using the points on the image
    l1_image = np.cross(points[2][0], points[2][1])  # l1 perpendicular to l2
    l2_image = np.cross(points[2][2], points[2][3])
     
    l3_image = np.cross(points[3][0], points[3][1]) # l3 perpendicular to l4
    l4_image = np.cross(points[3][2], points[3][3])

    # Lines after affine correction
    l1_affine = np.dot(np.linalg.inv(Ha).T, l1_image) 
    l2_affine = np.dot(np.linalg.inv(Ha).T, l2_image)
    l3_affine = np.dot(np.linalg.inv(Ha).T, l3_image) 
    l4_affine = np.dot(np.linalg.inv(Ha).T, l4_image) 
    
    x1, y1, z1 = l1_affine
    x2, y2, z2 = l2_affine
    x3, y3, z3 = l3_affine
    x4, y4, z4 = l4_affine

    A = np.array([[x1*x2, x1*y2+x2*y1, y1*y2], [x3*x4, x3*y4+x4*y3, y3*y4]])

    u, s, v = np.linalg.svd(A)
    p, q, r = v[-1]

    M = np.array([[p,q],[q,r]])

    if not is_pos_def(M):
        M = -M

    if is_pos_def(M):
        K = np.linalg.cholesky(M)
        Hm_inverted = np.array([[K[0][0],K[0][1],0],[K[1][0],K[1][1],0],[0,0,1]])
        Hm = np.linalg.inv(Hm_inverted)

    else:
        sys.exit('Matrix for Cholesky decomposition is not positive definite. Hm cannot be determined.')

    return Hm

def rectifyImage(img, debug, img_name):
    
    if debug:
        no_line_pairs = 2
        pts = []

        print('Select two sets of parallel lines')
        for i in range(no_line_pairs):
            print("For {} set: Select two points on a line along its direction".format(i+1))
            plt.imshow(img)
            x = plt.ginput(4, timeout = 0, show_clicks=True)
            x = np.array(x)
            x = np.insert(x, 2, 1, axis = 1)
            pts.append(x)
            plt.close()
        
        print('Select two sets of perpendicular lines')
        for i in range(no_line_pairs):
            print("For {} set: Select two points on a line along its direction".format(i+1))
            plt.imshow(img)
            x = plt.ginput(4, timeout = 0, show_clicks=True)
            x = np.array(x)
            x = np.insert(x, 2, 1, axis = 1)
            pts.append(x)
            plt.close()

        np.savez('./data/{}.npz'.format(img_name), **{img_name: pts})

    if not debug:
        
        points = np.load('./data/{}.npz'.format(img_name))
        Ha = affine_rectification(points[img_name])
        h, w, _ = img.shape

        img_corner_coords = np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]])
        img_corner_coords = img_corner_coords.T 
        edge_coords_matrix = np.dot(Ha, img_corner_coords)
        
        edge_coords_matrix = edge_coords_matrix/edge_coords_matrix[-1]
        max_h, max_w, _ = np.max(edge_coords_matrix, axis = 1)
        min_h, min_w, _ = np.min(edge_coords_matrix, axis = 1)
        
        affine_img = cv2.warpPerspective(img, Ha, (int(max_h - min_h), int(max_w - min_w)))
        
        plt.imshow(affine_img)
        plt.show()

        Hm = metric_rectification(Ha, points[img_name])

        edge_coords_matrix = np.dot(Hm, edge_coords_matrix)
        
        edge_coords_matrix = edge_coords_matrix/edge_coords_matrix[-1]
        max_h, max_w, _ = np.max(edge_coords_matrix, axis = 1)
        min_h, min_w, _ = np.min(edge_coords_matrix, axis = 1)

        metric_img = cv2.warpPerspective(affine_img, Hm, (int(max_h - min_h), int(max_w - min_w)))

        plt.imshow(metric_img)
        plt.show()
        
def main():

    # Load all the images that has to be metrically rectified
    img_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW1/HW1/images'
    images = glob.glob(img_folder+'/*')
    # for img in images:
    img_name = images[2].split('/')[-1]
    img = imread(images[2])
    # rectifyImage(img, 1, img_name)
    rectifyImage(img, 0, img_name)


if __name__ == "__main__":
    main()