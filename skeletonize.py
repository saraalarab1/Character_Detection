import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import cv2
# Invert the horse image
def skeletonize_image(image, image_name):
    """
    This function skeletonize an image and put it in a folder called skeletonized
    """
    # image = cv2.imread('Img_2/img006-001.png')
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    image = cv2.resize(image, (40, 40), interpolation = cv2.INTER_AREA)
    black_pixels = np.where(
        (image[:, :, 0] == 0) & 
        (image[:, :, 1] == 0) & 
        (image[:, :, 2] == 0)
    )
    white_pixels = np.where(
        (image[:, :, 0] == 255) & 
        (image[:, :, 1] == 255) & 
        (image[:, :, 2] == 255)
    )
    # set those pixels to white
    image[black_pixels] = [255, 255, 255]
    image[white_pixels] = [0, 0, 0]

    # perform skeletonization
    image = skeletonize(image)
    green_pixels = np.where(
        (image[:, :, 0] == 0) & 
        (image[:, :, 1] == 255) & 
        (image[:, :, 2] == 0)
        )
    black_pixels = np.where(
        (image[:, :, 0] == 0) & 
        (image[:, :, 1] == 0) & 
        (image[:, :, 2] == 0)
        )
    image[green_pixels] = [0, 0, 0]
    image[black_pixels] = [255, 255, 255]
    # display results
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
    #                         sharex=True, sharey=True)

    # ax = axes.ravel()

    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('original', fontsize=20)

    # ax[1].imshow(skeleton, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('skeleton', fontsize=20)

    # fig.tight_layout()
    # plt.show()
    cv2.imwrite('skeletonized_images_2/'+image_name, image)

# skeletonize_image('image', 'sldkfjdl')