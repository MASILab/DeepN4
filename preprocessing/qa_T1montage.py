import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt

with open('all_T1s.txt', 'r') as f:
    a = f.read().splitlines()
a.sort()

def plot_montage(few_files,ax):
    x = 0
    y = 0 
    for z in range(len(few_files)):
        image = nib.load(few_files[z]).get_fdata()
        #image = image[:,:,np.int32(image.shape[2] / 2)]
        image = image[:,:,np.int32(image.shape[2] / 1.7)]
        ax_image = np.squeeze(image)
        ax_image = np.rot90(ax_image,1)
        ax[x,y].imshow(ax_image, cmap='gray')
        ax[x,y].set_title(z)
        ax[x,y].axis('off')
        if y == 4:
            x += 1
        y += 1
        if y != 0 and y % 5 == 0:
            y = 0

i = 10400#0
j = 10400+25
while j < 16531:
    print('lala')
    few_files = a[i:j]
    fig, ax = plt.subplots(5,5,figsize=(10,20))
    fig.suptitle('Starts at: ' + str(i))
    plot_montage(few_files,ax)
    plt.show(block=False)
    plt.pause(0.02)
    plt.close()
    i = i + 25
    j = j + 25


