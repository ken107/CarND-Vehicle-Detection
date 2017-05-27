import matplotlib.image as mpimg
import numpy as np
import pickle
from glob import glob
from time import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_extraction import convert_color, grid


def train(color_space, pixels_per_cell, cells_per_block, hog_orientations, color_hist_nbins, window_shape):
    cars = glob("../download/vehicles/**/*.png")
    notcars = glob("../download/non-vehicles/**/*.png")
    files = cars + notcars

    features = []
    for file in files:
        img = mpimg.imread(file)
        img = convert_color(img, color_space)
        grid_shape, get_window = grid(img, pixels_per_cell, cells_per_block, hog_orientations, color_hist_nbins)
        window_bbox, img_features = get_window(cell_x=0, cell_y=0, window_shape=window_shape, is_complete_image=True)
        features.append(img_features)
        if len(features) % 1000 == 0: print("Extracting features: {}/{}\r".format(len(features), len(files)))

    # Feature vector
    X = np.float64(features)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
    assert(len(scaled_X) == len(y))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', hog_orientations, 'orientations', pixels_per_cell, 'pixels per cell and', cells_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t=time()
    svc.fit(X_train, y_train)
    t2 = time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Save
    with open("model.p", "wb") as f:
        pickle.dump([color_space, pixels_per_cell, cells_per_block, hog_orientations, color_hist_nbins, window_shape, svc, X_scaler], f)
        print('Model saved')



if "__main__" == __name__:
    train("YCrCb", 8, 2, 9, 16, (8,8))
