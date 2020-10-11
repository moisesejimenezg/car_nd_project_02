import matplotlib.pyplot as plt
import numpy as np

from color import Color

img = plt.imread("../test_images/test1.jpg")
color = Color(img)

filtered = color.Filter((170, 255))

plt.imshow(filtered, cmap="gray")
plt.show()
