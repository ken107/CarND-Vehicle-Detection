import cv2
import numpy as np
from skimage.feature import hog


def convert_color(img, color_space):
    if color_space == 'RGB':
        feature_image = np.copy(img)
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return feature_image


def grid(img, pixels_per_cell, cells_per_block, hog_orientations, color_hist_nbins):
    # ensure normalized image
    assert(img.dtype == np.float32)

    # compute shape & block-shape
    shape = [
        img.shape[0] // pixels_per_cell,
        img.shape[1] // pixels_per_cell
    ]
    nblocks = [
        shape[0] - cells_per_block + 1,
        shape[1] - cells_per_block + 1
    ]

    # compute hog for entire image
    hog_map = np.array([
        hog(img[:,:,0], orientations=hog_orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), block_norm='L1', transform_sqrt=True, feature_vector=False),
        hog(img[:,:,1], orientations=hog_orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), block_norm='L1', transform_sqrt=True, feature_vector=False),
        hog(img[:,:,2], orientations=hog_orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), block_norm='L1', transform_sqrt=True, feature_vector=False)
    ])

    # check hog shape
    assert(np.array_equal(hog_map.shape, [3, nblocks[0], nblocks[1], cells_per_block, cells_per_block, hog_orientations]))

    ## function to get a window inside the grid
    ## returns bounding box (pixels) and feature vector
    def get_window(cell_x, cell_y, window_shape, is_complete_image=False):
        # make sure know what we're doing
        if is_complete_image:
            assert(np.array_equal(window_shape, shape))

        # calculate number of blocks
        window_nblocks = [
            window_shape[0] - cells_per_block + 1,
            window_shape[1] - cells_per_block + 1
        ]

        # calculate bbox
        bbox = [
            (cell_x * pixels_per_cell, cell_y * pixels_per_cell),
            ((cell_x + window_shape[1]) * pixels_per_cell, (cell_y + window_shape[0]) * pixels_per_cell)
        ]

        # get image region inside window
        window_img = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

        # compute color histogram
        channel1_hist, edges = np.histogram(window_img[:,:,0], bins=color_hist_nbins, range=(0,1))
        channel2_hist, edges = np.histogram(window_img[:,:,1], bins=color_hist_nbins, range=(0,1))
        channel3_hist, edges = np.histogram(window_img[:,:,2], bins=color_hist_nbins, range=(0,1))

        # extract window's hog features
        hog_features = hog_map[:, cell_y:cell_y+window_nblocks[0], cell_x:cell_x+window_nblocks[1]].ravel()

        # feature vector
        features = np.concatenate([channel1_hist, channel2_hist, channel3_hist, hog_features])
        return bbox, features

    # return grid's shape and get_window function
    return shape, get_window
