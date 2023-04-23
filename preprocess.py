import numba as nb
import cv2
import numpy as np

@nb.njit(fastmath=True)
def contrast_img(img, colours):
    """
    Return the high-contrast image,
    with each pixel set as the closest high-contrast colour.
    """
    rgb_img = np.copy(img)
    for i, row in enumerate(rgb_img):
        for j, rgb_pixel in enumerate(row):
            dists = np.empty(len(colours))
            for k, colour in enumerate(colours):
                dist = 0
                for x, y in zip(rgb_pixel, colour):
                    dist += np.abs(x-y) ** 2
                dists[k] = dist ** (1/2)
            min_val = dists[0]
            min_ind = 0
            for l in range(1, len(colours)):
                if dists[l] < min_val:
                    min_val = dists[l]
                    min_ind = l
            rgb_img[i,j] = colours[min_ind]
    return rgb_img


@nb.njit(fastmath=True)
def rgb_to_grey(img):
    """
    Convert an RGB image to greyscale using the weighted method.
    """
    num_rows, num_cols, _ = img.shape
    grey_img = np.empty((num_rows, num_cols), dtype=np.uint8)
    for i, row in enumerate(img):
        for j, rgb_pixel in enumerate(row):
            # Compute weighted sum of RGB channels
            grey_img[i, j] = 0.2989 * rgb_pixel[0] + 0.5870 * rgb_pixel[1] + 0.1140 * rgb_pixel[2]
    return grey_img


def process_img(img, crop="box", pool_size=2, contrast=True, greyscale=True, normalise=True):
    """
    Pre-process the image
    """
    if crop == "box":
        # crop unnecessary pixels
        img = img[12:-12, 12:-12]
        
    if pool_size is not None:
        # Average pooling according to pool size
        img_shape = img.shape
        img = cv2.resize(img.astype("float32"), (img_shape[0]//pool_size, img_shape[1]//pool_size), interpolation=cv2.INTER_AREA).astype(np.uint8)
        
    if contrast:
        # Set each pixel colour as its closest high-contrast colour
        colours = np.array([[170,0,0],[105,230,105],[0,0,0],[101,101,101],[255,255,255]])
        img = contrast_img(img, colours)
    
    if greyscale:
        # Convert the image to greyscale
        img = rgb_to_grey(img)
    
    if normalise:
        # Change pixel intensity scale to [0,1]
        img = img.astype(np.float32) / 255
        
    return img