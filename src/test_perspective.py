import matplotlib.pyplot as plt

from geometry import Point
from geometry import Trapezoid
from geometry import Transformation
from perspective import Perspective

src_p0 = Point(200, 720)
src_p1 = Point(570, 460)
src_p2 = Point(725, 460)
src_p3 = Point(1115, 720)

src_trp = Trapezoid(src_p0, src_p1, src_p2, src_p3)

dst_p0 = Point(200, 720)
dst_p1 = Point(200, 460)
dst_p2 = Point(1000, 460)
dst_p3 = Point(1000, 720)

dst_trp = Trapezoid(dst_p0, dst_p1, dst_p2, dst_p3)

transformation = Transformation(src_trp, dst_trp)
perspective = Perspective(transformation)

img = plt.imread('../test_images/straight_lines1.jpg')

transformed = perspective.transform(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img, cmap='gray')
ax1.set_title('Original', fontsize=50)
ax2.imshow(transformed, cmap='gray')
ax2.set_title('Transformed', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
