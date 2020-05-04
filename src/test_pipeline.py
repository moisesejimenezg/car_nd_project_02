import cv2
import matplotlib.pyplot as plt

from pipeline import Pipeline

pipeline = Pipeline(9, 6)
pipeline.Calibrate('../camera_cal/calibration*.jpg')
pipeline.Undistort('../test_images/*.jpg')

plt.imshow(pipeline.images_[0])
plt.show()
