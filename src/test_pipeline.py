import cv2
import matplotlib.pyplot as plt

from geometry import Point
from geometry import Transformation
from geometry import Trapezoid
from pipeline import Pipeline

pipeline = Pipeline(9, 6, 15)
pipeline.Calibrate('../camera_cal/calibration*.jpg')
img = plt.imread('../test_images/straight_lines1.jpg')
pipeline.Undistort(img)
pipeline.CalculateGradient()
pipeline.FilterGradients((20,100), (30, 100), (0.7, 1.3))
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

f, (ax1, ax2) = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()
ax1[0].imshow(img)
ax1[0].set_title('Raw', fontsize=10)
ax1[1].imshow(pipeline.img_, cmap='gray')
ax1[1].set_title('Undistorted', fontsize=10)
ax2[0].imshow(combinedA, cmap='gray')
ax2[0].set_title('Combined', fontsize=10)
ax2[1].imshow(transformed, cmap='gray')
ax2[1].set_title('Transformed', fontsize=10)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
