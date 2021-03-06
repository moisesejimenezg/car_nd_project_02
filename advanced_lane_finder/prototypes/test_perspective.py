import matplotlib.pyplot as plt

from advanced_lane_finder.core.geometry import Point
from advanced_lane_finder.core.geometry import Transformation
from advanced_lane_finder.core.geometry import Trapezoid
from advanced_lane_finder.core.perspective import Perspective

src_p0 = Point(165, 720)
src_p1 = Point(550, 480)
src_p2 = Point(730, 480)
src_p3 = Point(1115, 720)

src_trp = Trapezoid(src_p0, src_p1, src_p2, src_p3)

dst_p0 = Point(400, 720)
dst_p1 = Point(400, 460)
dst_p2 = Point(880, 460)
dst_p3 = Point(880, 720)

dst_trp = Trapezoid(dst_p0, dst_p1, dst_p2, dst_p3)

transformation = Transformation(src_trp, dst_trp)
perspective = Perspective(transformation)

img = plt.imread("advanced_lane_finder/data/test_images/straight_lines1.jpg")

transformed = perspective.Transform(img)

f, (ax1, ax2) = plt.subplots(1, 2)
f.tight_layout()
ax1.imshow(img, cmap="gray")
ax1.set_title("Original")
ax2.imshow(transformed, cmap="gray")
ax2.set_title("Transformed")
plt.show()
