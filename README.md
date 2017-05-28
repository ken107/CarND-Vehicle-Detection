# Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

**Project Files**

- _Main.ipynb_: IPython notebook defining the pipeline, parameter values, training, testing, and video processing
- _feature_extraction.py_: module containing code dealing with feature extraction
- _training.py_: module containing code to load training data and train a linear SVM classifier
- _window_search.py_: module containing code to search for cars in an image using sliding windows

[//]: # (Image References)
[image1]: ./output_images/cars.png
[image8]: ./output_images/notcars.png
[image2]: ./output_images/car_hog.png
[image9]: ./output_images/notcar_hog.png
[image10]: ./output_images/car_hist.png
[image11]: ./output_images/notcar_hist.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/detected_vehicles.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting HOG features is in _feature_extraction.py line 37_.  In the function `grid`, I compute HOG features for the entire image, then return a lambda that will return HOG features for specific windows.  The `grid` function is called from _training.py line 21_ for each image in the training dataset.

The training data consists of vehicle and non-vehicle images.  Here is a few examples of the training images:

![alt text][image1]

![alt text][image8]

#### 2. Explain how you settled on your final choice of HOG parameters.

The training parameters are defined in the first code cell in _Main.ipynb_.

I tried a few combinations of `orientation`, `pixels_per_cell`, and `cells_per_block`.  I didn't notice significant difference in performance.  The default settings of `orientation=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)` appeared to work fairly well.  So I went with that.  I also went with YCrCb after trying out various color spaces.

Using the `vis` parameter of `skimage.hog()`, I visualized the HOG features of a few images to get an idea of how the method works.

![alt text][image2]
![alt text][image9]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I also experimented with color spatial and color histogram features.  I found that color spatial features didn't work well, so I did not incorporate it.  But color histogram features appeared to help classification, though the number of bins didn't seem to affect the result much.  I went with `nbins=16`, see _feature_extraction.py line 69_.

And here are color histograms for the previous images:

![alt text][image10]
![alt text][image11]

After extracting features for all training images, I create a corresponding array of labels: 1 for vehicles, 0 for non-vehicles.  Then I shuffle and split the data into training and test sets (80/20).  And finally I train a Linear SVM Classifier and validate using these datasets (see _training.py line 32-42_).  I also save the parameter values and the trained model to a file named _model.p_.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My sliding window code can be found in _window_search.py line 8-29_.  I begin by cropping the image, keeping only the lower half.  Then I resize it by the specified scaling factor, and then convert to YCrCb.  I decided to use just one scale factor, 2/3, corresponding to window size (96,96).  I find this is sufficient for the current project.

After scaling & color conversion, I call the `grid` function from _feature_extractor.py_ to divide the image into a grid of cells.  Then I slide a 8x8-cell (64x64-pixel) window over the image, with `cells_per_step=(2,2)` (i.e. stepping 2 cells at a time).  With each step, I extract the HOG/hist features of the window and feed it into the classifier (_window_search.py line 26_).  If the result is a vehicle, I return that window's bounding box.

In the following image, I drew the slided windows to visualize its coverage:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images of the detected vehicles, albeit with false positives:

![alt text][image4]

As I mentioned earlier, the main optimization is by calculating HOG features for the entire image to avoid duplicate HOG calculations for overlapping window regions.  This is accomplished by the `grid` function in _feature_extraction.py_.  Doing this allows having significant window overlap without incurring extra HOG computation cycles.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To deal with false positives, I save the bounding boxes of the previous 24 video frames, and stack them to create a heatmap (see the 2nd code cell of _Main.ipynb_).  Then I threshold (>=22) the heatmap and use `scipy.ndimage.measurements.label()` to identify blobs that represent vehicles, and calculate the bounding boxes of the blobs.

Here's a short video showing the heatmap at work.  Notice a few "cool" detections appear briefly on the heatmap, but they're filtered out by the threshold, such that they don't appear in the output on the right.

[heatmap video](./test_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially I tried to use some of the code provided in the lecture.  But due to the complexity of the algorithm involved, especially the confusion with HOG cells and blocks and windows, it was difficult to understand and adapt the provided code.  So at one point, I decided to rewrite everything from the ground up and modularize to make it easier to manage.

One particular challenge was with false positives, and finding a good heatmap accumulation strategy and a threshold to filter them out.  I thought about keep previous heatmaps, but worried about memory usage.  In the end, I figured I could simply keep the list of bounding boxes from previous frames and that would accomplish the same goal.

I tried to think of ways to stop the wobbliness, perhaps by averaging the final bounding boxes over multiple consecutive frames, but it implied correlating these bboxes across frames, which doesn't seem too difficult, but is nevertheless non-trivial.  I did not pursue it.

I think my pipeline will fail when vehicles don't match the data set, such as trucks carrying stuff, big rigs, candy-painted cars, exotic cars.  It may also fail if the car is too close or too long, and the windows aren't big enough to capture it.  More data certainly helps, and bigger windows.

But I also thought about why we don't use two cameras instead of one to detect depth and use the depth information to help with object detection.  After all it is how our eyes work.  Detecting depth from stereocopic images is tricky because objects appear slightly different in the two images, but OpenCV already provides a method `createStereoBM` to do it.  Depth information would make vehicle detection significantly more effective.
