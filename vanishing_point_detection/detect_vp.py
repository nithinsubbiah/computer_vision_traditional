import scipy.io
import glob
import cv2
import numpy as np 

def cluster_lines(lines, bin_size = 10):

	# To avoid the problem of dividing by zero, 1e-5 is added to the denominator
	slope = (lines[:,3]-lines[:,2])/((lines[:,1]-lines[:,0])+1e-5)
	theta = np.arctan(slope)
	sorted_idx = np.argsort(theta)
	theta = np.sort(theta)
	histogram_values, bin_edges = np.histogram(theta, bins = bin_size, range = (-np.pi/2,np.pi/2))

	sorted_lines = lines[sorted_idx]

	clusters = []
	# Variable to keep count of how many lines has been plotted
	line_count = 0
	for i in range(bin_size):
		clusters.append(sorted_lines[line_count:line_count+histogram_values[i]])
		line_count += histogram_values[i]

	return clusters

def show_clusters(image, clusters):

	'''
	Input: 1. Image on which the cluster of lines is to be plotted
		   2. A list of list of numpy arrays belonging to each cluster
	Output: Shows clusters on the image
	'''

	for cluster in clusters:
		# Unique color for each cluster
		random_nos = np.random.randint(low=0, high=255, size=3)
		for line in cluster:
			x1, x2, y1, y2, _, _ = line
			cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), [int(255-random_nos[0]), int(255-random_nos[1]), int(255-random_nos[2])], 2) 	

	cv2.imshow('img', image)
	cv2.waitKey(0)

def main():
	img_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/input'
	lines_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/lines'

	images = glob.glob(img_folder+'/*')

	image = images[-5]
	img = cv2.imread(image)

	img_name = (image.split('/')[-1]).split('.')[0]

	# lines are of the structure (x1,x2,y1,y2,theta,r)
	lines = scipy.io.loadmat(lines_folder+'/'+img_name+'.mat')['lines']

	# histogram_values gives the number of sorted_lines in each bin.  
	clusters = cluster_lines(lines)

	# show_clusters(img, clusters)
	
	

	'''

	for line in lines:
		x1, x2, y1, y2, _, _ = line 
		cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 2) 
	

	cv2.imshow(img_name, img)
	cv2.waitKey(0)
	'''


if __name__ == '__main__':
	main()