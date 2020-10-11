import cv2
import matplotlib.pyplot as plt

from geometry import Point
from geometry import Transformation
from geometry import Trapezoid
from pipeline import Pipeline

pipeline = Pipeline(9, 6, 15)
pipeline.Prepare()
img = plt.imread("../test_images/test3.jpg")
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
pipeline.CalculateCurvature(left_fit.rw_polynomial_, y_eval)
pipeline.CalculateCurvature(right_fit.rw_polynomial_, y_eval)

print("Left curvature: " + str(pipeline.GetCurvatures()["left"]))
print("Right curvature: " + str(pipeline.GetCurvatures()["right"]))

f, (ax1, ax2) = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()
ax1[0].imshow(pipeline.img_)
ax1[0].set_title("Undistorted", fontsize=10)
ax1[1].imshow(combinedA, cmap="gray")
ax1[1].set_title("Combined", fontsize=10)
ax2[0].imshow(transformed, cmap="gray")
ax2[0].set_title("Transformed", fontsize=10)
ax2[1].imshow(result)
ax2[1].set_title("Fitted", fontsize=10)
plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.0)
plt.show()
