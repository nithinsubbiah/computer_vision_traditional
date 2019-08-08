import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2 as cv
import LucasKanade

# write your script here, we recommend the above libraries for making your animation

def main():

    frames = np.load('../data/carseq.npy')

    fig, ax = plt.subplots(1)
    rect = np.array((59,116,145,151))
    template0 = frames[rect[1]:rect[3]+1,rect[0]:rect[2]+1,0]

    rect_save = []


    for i in range(frames.shape[-1]-1):

        p0 = LucasKanade.LucasKanade(template0,frames[:,:,i+1],rect)

        template = frames[rect[1]:rect[3]+1,rect[0]:rect[2]+1,i]
        p1 = LucasKanade.LucasKanade(template,frames[:,:,i+1],rect)

        if(np.linalg.norm(p0-p1)>3):
            rect = np.round(rect + np.vstack([p1,p1]).reshape(-1,)).astype(int)
        else:
            rect = np.round(rect + np.vstack([p0,p0]).reshape(-1,)).astype(int)

        w = rect[2]-rect[0]
        h = rect[3]-rect[1]

        BB = patches.Rectangle((rect[0],rect[1]),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(BB)
        plt.imshow(frames[:,:,i+1],cmap='gray')
        plt.pause(0.0001)

        rect_save.append(rect)
        if i in [1,100,200,300,400]:
            plt.savefig('carseq_'+ str(i)+'.png')

        BB.remove()

    rect_save = np.vstack(rect_save)
    np.save('carseqrects-wcrt.npy',rect_save)

if __name__ == '__main__':
    main()
