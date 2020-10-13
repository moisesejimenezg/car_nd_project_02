import matplotlib.pyplot as plt
import numpy as np

from advanced_lane_finder.core.color import Color

img = plt.imread("advanced_lane_finder/data/test_images/test1.jpg")
color = Color(img)

filtered = color.Filter((170, 255))

f, (ax1, ax2) = plt.subplots(1, 2)
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Raw image')
ax2.imshow(filtered)
ax2.set_title('Color Filtered Image')
plt.show()
