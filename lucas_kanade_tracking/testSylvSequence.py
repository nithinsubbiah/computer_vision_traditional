import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanadeBasis
import LucasKanade

# write your script here, we recommend the above libraries for making your animation

def main():

    frames = np.load('../data/sylvseq.npy')
    bases = np.load('../data/sylvbases.npy')
    fig, ax = plt.subplots(1)
    rect = np.array([101,61,155,107])
    template0 = frames[rect[1]:rect[3]+1,rect[0]:rect[2]+1,0]

    rect_save = []

    for i in range(frames.shape[-1]-1):

        template = frames[rect[1]:rect[3]+1,rect[0]:rect[2]+1,i]
        p = LucasKanadeBasis.LucasKanadeBasis(template,frames[:,:,i+1],rect,bases)
        #p = LucasKanade.LucasKanade(template,frames[:,:,i+1],rect,bases)
        rect = np.round(rect + np.vstack([p,p]).reshape(-1,)).astype(int)
        w = rect[2]-rect[0]
        h = rect[3]-rect[1]

        BB = patches.Rectangle((rect[0],rect[1]),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(BB)
        plt.imshow(frames[:,:,i+1],cmap='gray')
        plt.pause(0.0001)

        rect_save.append(rect)
        if i in [1,200,300,350,400]:
            plt.savefig('sylvseq_'+ str(i)+'.png')

        BB.remove()

    rect_save = np.vstack(rect_save)
    np.save('sylvseqrects.npy',rect_save)

if __name__ == '__main__':
    main()
