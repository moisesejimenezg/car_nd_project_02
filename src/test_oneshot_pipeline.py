import cv2
import matplotlib.pyplot as plt

from geometry import Point
from geometry import Transformation
from geometry import Trapezoid
from pipeline import Pipeline

pipeline = Pipeline(9, 6, 15)
pipeline.Calibrate("../camera_cal/calibration*.jpg")
img = plt.imread("../test_images/test3.jpg")

result, left_curvature, right_curvature = pipeline.Process(img)

print("Left curvature: " + str(left_curvature))
print("Right curvature: " + str(right_curvature))

plt.imshow(result)
plt.show()
