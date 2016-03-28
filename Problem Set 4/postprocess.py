
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():

	# Given output data from the CT reconstruction,
	# produce a grayscale image (PNG format)
    image_data = np.loadtxt(sys.argv[1])
    plt.figure(0)
    plt.imshow(image_data, interpolation='nearest', cmap = cm.Greys_r)

    fname_reconstruct_image = sys.argv[1] + "_image.png"
    plt.savefig(fname_reconstruct_image)


if __name__ == '__main__':
    main()
