import matplotlib.pyplot as plt
import numpy as np

from advanced_lane_finder.core.color import Color

img = plt.imread("advanced_lane_finder/data/test_images/test1.jpg")
color = Color(img)

filtered = color.Filter((170, 255))

plt.imshow(filtered, cmap="gray")
plt.show()
