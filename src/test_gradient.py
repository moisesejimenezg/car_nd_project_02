import matplotlib.pyplot as plt

from gradient import Gradient

gradient = Gradient(15)
img = plt.imread('../test_images/signs_vehicles_xygrad.png')
gradient.CalculateGradient(img)

abs_threshold = gradient.AbsoluteThreshold((20,100))
# plt.imshow(abs_threshold[0], cmap='gray')
# plt.show()

mag_threshold = gradient.MagnitudeThreshold((30, 100))
# plt.imshow(mag_threshold, cmap='gray')
# plt.show()

dir_threshold = gradient.DirectionThreshold((0.7, 1.3))
plt.imshow(dir_threshold, cmap='gray')
plt.show()
