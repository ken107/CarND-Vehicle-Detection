import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from moviepy.editor import VideoFileClip
from window_search import *


with open("model.p", "rb") as f:
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, spatial_feat, hist_feat, hog_feat, y_start_stop, svc, X_scaler = pickle.load(f)

def draw_boxes(img, bboxes, color, thick):
    for bbox in bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

def process(image):
    assert(image.dtype == np.uint8 and hist_range == (0,1))
    image = image.astype(np.float32)/255

    windows = slide_window(image,
        x_start_stop=[None, None],
        y_start_stop=y_start_stop,
        xy_window=(96, 96),
        xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        hist_range=hist_range,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        spatial_feat=spatial_feat,
        hist_feat=hist_feat,
        hog_feat=hog_feat)

    draw_boxes(image, hot_windows, color=(0, 0, 1), thick=6)
    return (image*255).astype(np.uint8)


clip1 = VideoFileClip("../../../Desktop/carnd/temp/project_video.mp4")
clip2 = clip1.subclip(10,15).fl_image(process)
clip2.write_videofile("R:/project_video.mp4", audio=False)
#image = clip1.get_frame(20)
#window_img = process(image)

#plt.imshow(window_img)
#plt.show()
