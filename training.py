import numpy as np
import pickle
from glob import glob
from time import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features


def train(sample_size, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, spatial_feat, hist_feat, hog_feat):
    # Read in cars and notcars
    cars = glob("../download/vehicles/**/*.png")
    notcars = glob("../download/non-vehicles/**/*.png")

    # Reduce the sample size
    if sample_size is not None:
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
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
    print('Saving model.p')
    with open("model.p", "wb") as f:
        pickle.dump([color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, spatial_feat, hist_feat, hog_feat, svc, X_scaler], f)
