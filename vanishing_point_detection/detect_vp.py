import scipy.io
import glob
import cv2
import numpy as np 

def cluster_lines(lines, bin_size = 10):
	'''
	Input: 1. Lines that needs to be clustered
		   2. Size of the bin to be used in histogram
	Output: List of list of clustered lines 
	'''

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
	
	Output: None

	Description: Displays clusters on the image
	'''

	for cluster in clusters:
		# Unique color for each cluster
		random_nos = np.random.randint(low=0, high=255, size=3)
		for line in cluster:
			x1, x2, y1, y2, _, _ = line
			cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), [int(255-random_nos[0]), int(255-random_nos[1]), int(255-random_nos[2])], 2) 	

	cv2.imshow('img', image)
	cv2.waitKey(0)

def RANSAC(cluster, iterations = 1000, threshold = 0.1):
	
	max_inliers = 0

	for i in range(iterations):
		
		num_inliers = 0
		data_1, data_2 = cluster[np.random.randint(len(cluster),size=2)]
		line_1 = np.cross([data_1[0],data_1[2],1],[data_1[1],data_1[3],1])
		line_2 = np.cross([data_2[0],data_2[2],1],[data_2[1],data_2[3],1])
		VP = np.cross(line_1, line_2)

		for data in cluster:
			line = np.cross([data[0],data[2],1],[data[1],data[3],1])
			# Error metric: distance of the VP from each line
			distance = np.dot(line, VP)/np.sqrt(line[0]**2 + line[1]**2)
			
			if distance > threshold:
				num_inliers += 1
		
		if num_inliers > max_inliers:


def refine_clusters(clusters):

	for cluster in clusters:
		RANSAC(cluster)

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
	refine_clusters(clusters)

	


if __name__ == '__main__':
	main()