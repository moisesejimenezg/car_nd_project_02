import cv2
import matplotlib.pyplot as plt

from advanced_lane_finder.core.geometry import Point
from advanced_lane_finder.core.geometry import Transformation
from advanced_lane_finder.core.geometry import Trapezoid
from advanced_lane_finder.core.pipeline import Pipeline

pipeline = Pipeline(9, 6, 15)
pipeline.Prepare()

img = plt.imread("./advanced_lane_finder/data/test_images/test3.jpg")

result = pipeline.Process(img)

plt.imshow(result)
plt.show()
