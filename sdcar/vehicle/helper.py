import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
import random

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

if __name__ == "__main__":
    # Display 10 vehicle images
    vehicles = glob.glob("../../vehicles/KITTI_extracted/*.png")
    nonvehicles = glob.glob("../../non-vehicles/Extras/extra*.png")
    n = random.randint(0,1000)
    
    fnames = vehicles[n:n+3] + nonvehicles[n:n+3]
    
    # Select a small fraction of pixels to plot by subsampling it
    for fname in fnames:
        img = cv2.imread(fname)
        

        # Select a small fraction of pixels to plot by subsampling it
        scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
        img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

        # Convert subsampled image to desired color space(s)
        img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
        # OpenCV uses BGR, matplotlib likes RGB
        img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2LUV)
        img_small_YUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2YUV)
        img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_BGR2HLS)

        img_small_rgb = img / 255.  # scaled to [0, 1], only for plotting

        # Plot and show
        plt.imshow(img_small_RGB)
        plt.title('image')
        plt.show()

        plot3d(img_small_RGB, img_small_rgb)
        plt.title(fname + "-RGB")
        plt.show()

        plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
        plt.title(fname + "-HSV")
        plt.show()

        plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
        plt.title(fname + "-LUV")
        plt.show()

        plot3d(img_small_YUV, img_small_rgb, axis_labels=list("YUV"))
        plt.title(fname + "-YUV")
        plt.show()

        plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
        plt.title(fname + "-HLS")
        plt.show()
