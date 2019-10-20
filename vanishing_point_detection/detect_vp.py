import scipy.io
import glob
import cv2
import numpy as np 
import matplotlib.pyplot as plt

def cluster_lines(lines, bin_size=10):
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
    histogram_values, bin_edges = np.histogram(theta, bins=bin_size, range=(-np.pi/2,np.pi/2))

    sorted_lines = lines[sorted_idx]

    clusters = []
    # Variable to keep count of how many lines has been plotted
    line_count = 0
    
    for i in range(bin_size):
        clusters.append(sorted_lines[line_count:line_count + histogram_values[i]])
        line_count += histogram_values[i]

    return clusters

def show_clusters(image, clusters, points=None, show_point=False):

    '''
    Input: 1. Image on which the cluster of lines is to be plotted
           2. A list of list of numpy arrays belonging to each cluster
           3. Optional: Vanishing point corresponding to every cluster
           4. Optional: Bool whether to display the point
    
    Output: None

    Description: Displays clusters on the image
    '''

    # Unique color for each cluster. (BGR)
    #Assuming only five clusters are formed
    line_color = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,255,0)]

    for idx, cluster in enumerate(clusters):
        
        for line in cluster:
            x1, x2, y1, y2, _, _ = line
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color=line_color[idx], thickness=2)    

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if show_point:
        VP = np.array(points)
        VP = np.transpose(VP.T/VP[:,-1])
        
        # Point colors corresponding to that of lines
        point_colors = ['b', 'g', 'r', 'y', 'c']

        for idx, point in enumerate(VP):
            plt.scatter(int(point[0]), int(point[1]), c=point_colors[idx])

    plt.show()


def RANSAC(cluster, iterations=5000, threshold=0.35):
    
    max_inliers = 0
    best_VP = None

    print("Running RANSAC...")

    for i in range(iterations):
        num_inliers = 0
        data_1, data_2 = cluster[np.random.randint(len(cluster),size=2)]
        line_1 = np.cross([data_1[0],data_1[2],1],[data_1[1],data_1[3],1])
        line_2 = np.cross([data_2[0],data_2[2],1],[data_2[1],data_2[3],1])
        VP = np.cross(line_1, line_2)
        
        for data in cluster:

            # Error metric: distance between end points of a line to the line joining VP and center of the end points
            center = np.array([(data[0]+data[1])/2,(data[2]+data[3])/2,1])
            line = np.cross(center, VP)
            distance = np.absolute(np.dot(line, np.array([data[0],data[2],1])))/np.linalg.norm(line[0:2]) 

            if distance < threshold:
                num_inliers += 1
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_VP = VP
    
    print("RANSAC complete")

    new_cluster = []

    for data in cluster:
        line = np.cross([data[0],data[2],1],[data[1],data[3],1])
        distance = np.dot(line, best_VP)/np.sqrt(line[0]**2 + line[1]**2)
        
        if distance < threshold:
            new_cluster.append(data)

    return best_VP, new_cluster

def find_VP(clusters):

    # Combining the bins with angle ranges (-90,-72) and (72,90)
    clusters[0] = np.concatenate((clusters[0], clusters[-1]))
    del clusters[-1]

    new_clusters = []
    VPs = []
    
    for cluster in clusters:
        
        # Discard the clusters that are too small
        if len(cluster) < 10:
            continue
        VP, new_cluster = RANSAC(cluster)
        VPs.append(VP)
        new_clusters.append(new_cluster)
    
    return new_clusters, VPs

def main():
    img_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/input'
    lines_folder = '/home/nithin/Desktop/Geometry Vision/Assignments/HW2/images/lines'

    images = glob.glob(img_folder+'/*')

    image = images[1]
    img = cv2.imread(image)

    img_name = (image.split('/')[-1]).split('.')[0]

    # lines are of the structure (x1,x2,y1,y2,theta,r)
    lines = scipy.io.loadmat(lines_folder+'/'+img_name+'.mat')['lines']

    # histogram_values gives the number of sorted_lines in each bin.  
    clusters = cluster_lines(lines)

    # show_clusters(img, clusters)
    new_clusters, VP = find_VP(clusters)

    show_clusters(img, new_clusters, points=VP, show_point=True)

if __name__ == '__main__':
    main()