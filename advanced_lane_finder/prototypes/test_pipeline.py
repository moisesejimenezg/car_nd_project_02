import cv2
import matplotlib.pyplot as plt

from advanced_lane_finder.core.geometry import Point
from advanced_lane_finder.core.geometry import Transformation
from advanced_lane_finder.core.geometry import Trapezoid
from advanced_lane_finder.core.pipeline import Pipeline

pipeline = Pipeline(9, 6, 15)
pipeline.Prepare()
img = plt.imread("advanced_lane_finder/data/test_images/test3.jpg")
pipeline.Undistort(img)
pipeline.CalculateGradient()
pipeline.FilterGradients((20, 100), (30, 100), (0.7, 1.3))
pipeline.InitColor()
pipeline.FilterColor((170, 255))

combinedA = pipeline.JoinOption("A")
combinedB = pipeline.JoinOption("B")
combinedC = pipeline.JoinOption("C")

transformed = pipeline.Transform(combinedA)

left_fit, right_fit = pipeline.FitPolynomial(transformed, False)

result = pipeline.PlotLaneOnImage(pipeline.img_, left_fit, right_fit)

y_eval = img.shape[0]
left_c = pipeline.CalculateCurvature(left_fit.rw_polynomial_, y_eval)
right_c = pipeline.CalculateCurvature(right_fit.rw_polynomial_, y_eval)
offset = pipeline.CalculateOffsetFromCenter(left_fit, right_fit)

print("Left curvature: " + str(left_c))
print("Right curvature: " + str(right_c))
print("Offset: " + str(offset))

f, (ax1, ax2) = plt.subplots(1, 2)
f.tight_layout()
ax1.imshow(pipeline.img_)
ax1.set_title("Undistorted", fontsize=10)
ax2.imshow(result)
ax2.set_title("Fitted", fontsize=10)
plt.show()
