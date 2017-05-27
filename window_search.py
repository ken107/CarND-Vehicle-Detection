import cv2
import numpy as np
from feature_extraction import convert_color, grid


def search_cars(img, y_start_stop, scale, color_space, pixels_per_cell, cells_per_block, hog_orientations, color_hist_nbins, window_shape, cells_per_step, svc, X_scaler):
    # get region of interest, scale, & convert color
    feature_img = img[y_start_stop[0]:y_start_stop[1]]
    feature_img = cv2.resize(feature_img, (0,0), fx=scale, fy=scale)
    feature_img = convert_color(feature_img, color_space)

    # gridify the image region
    grid_shape, get_window = grid(feature_img, pixels_per_cell, cells_per_block, hog_orientations, color_hist_nbins)

    # calculate number of windows
    nwindows = np.subtract(grid_shape, window_shape) // cells_per_step + 1

    # slide windows
    bboxes = []
    for cell_x in range(nwindows[1]):
        for cell_y in range(nwindows[0]):
            bbox, features = get_window(cell_x, cell_y, window_shape)
            test_features = X_scaler.transform(features.reshape(1, -1))
            if svc.predict(test_features):
                bboxes.append((bbox / scale).astype(np.uint8))

    return bboxes
