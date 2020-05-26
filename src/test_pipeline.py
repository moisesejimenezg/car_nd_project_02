import cv2
import matplotlib.pyplot as plt

from geometry import Point
from geometry import Transformation
from geometry import Trapezoid
from pipeline import Pipeline

pipeline = Pipeline(9, 6, 15)
pipeline.Calibrate('../camera_cal/calibration*.jpg')
img = plt.imread('../test_images/test3.jpg')
pipeline.Undistort(img)
pipeline.CalculateGradient()
pipeline.FilterGradients((20, 100), (30, 100), (0.7, 1.3))
pipeline.InitColor()
pipeline.FilterColor((170, 255))

combinedA = pipeline.JoinOption('A')
combinedB = pipeline.JoinOption('B')
combinedC = pipeline.JoinOption('C')

src_p0 = Point(190, 720)
src_p1 = Point(550, 480)
src_p2 = Point(730, 480)
src_p3 = Point(1115, 720)

src_trp = Trapezoid(src_p0, src_p1, src_p2, src_p3)

dst_p0 = Point(250, 720)
dst_p1 = Point(250, 460)
dst_p2 = Point(980, 460)
dst_p3 = Point(980, 720)

dst_trp = Trapezoid(dst_p0, dst_p1, dst_p2, dst_p3)

transformation = Transformation(src_trp, dst_trp)

pipeline.InitPerspective(transformation)
transformed = pipeline.Transform(combinedA)

left_fit, right_fit = pipeline.FitPolynomial(transformed, False)

result = pipeline.PlotLaneOnImage(pipeline.img_, left_fit, right_fit)

y_eval = img.shape[0]
left_curvature = pipeline.CalculateCurvature(left_fit.rw_polynomial_, y_eval)
right_curvature = pipeline.CalculateCurvature(right_fit.rw_polynomial_, y_eval)

print('Left curvature: ' + str(left_curvature))
print('Right curvature: ' + str(right_curvature))

f, (ax1, ax2) = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()
ax1[0].imshow(pipeline.img_)
ax1[0].set_title('Undistorted', fontsize=10)
ax1[1].imshow(combinedA, cmap='gray')
ax1[1].set_title('Combined', fontsize=10)
ax2[0].imshow(transformed, cmap='gray')
ax2[0].set_title('Transformed', fontsize=10)
ax2[1].imshow(result)
ax2[1].set_title('Fitted', fontsize=10)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
