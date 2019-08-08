import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

def main():

    frames = np.load('../data/aerialseq.npy')

    fig, ax = plt.subplots(1)
    rect_save = []

    for i in range(frames.shape[-1]-1):
        p = SubtractDominantMotion.SubtractDominantMotion(frames[:,:,i],frames[:,:,i+1])

        points = np.where(p)
        plt.plot(points[1],points[0],'g.')
        plt.imshow(frames[:,:,i],cmap='gray')
        plt.pause(0.0001)

        if i in [30,60,90,120]:
            plt.savefig('airseq_'+ str(i)+'.png')

        plt.clf()



if __name__ == '__main__':
    main()
