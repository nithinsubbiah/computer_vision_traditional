import numpy as np 
import glob
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2

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
    vanishing_line = vanishing_line/np.linalg.norm(vanishing_line)

    Ha = np.array([[1,0,0], [0,1,0], [vanishing_line[0],vanishing_line[1],vanishing_line[2]]])
    return Ha 

# def metric_rectification(Ha, points):


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
        
        np.savez('saved_points.npz', **{img_name: pts})

    if not debug:
        
        points = np.load('saved_points.npz')
        Ha = affine_rectification(points[img_name])
        h, w, _ = img.shape

        img_corner_coords = np.array([[0,0,1],[h,0,1],[0,w,1],[h,w,1]])
        img_corner_coords = img_corner_coords.T 
        edge_coords_matrix = np.dot(Ha, img_corner_coords)
        
        edge_coords_matrix = edge_coords_matrix/edge_coords_matrix[-1]
        max_h, max_w, _ = np.amax(edge_coords_matrix, axis = 1)
        affine_image = cv2.warpPerspective(img, Ha, (int(max_w), int(max_h)))
        plt.imshow(affine_image)
        plt.show()


def main():

    # Load all the images that has to be metrically rectified
    img_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW1/HW1/images'
    images = glob.glob(img_folder+'/*')
    # for img in images:
    img_name = images[0].split('/')[-1]
    img = imread(images[0])
    rectifyImage(img, 1, img_name)
    rectifyImage(img, 0, img_name)


if __name__ == "__main__":
    main()