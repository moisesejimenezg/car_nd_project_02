import matplotlib.pyplot as plt
import numpy as np

from gradient import Gradient

gradient = Gradient(3)
img = plt.imread('../test_images/test1.jpg')
gradient.CalculateGradient(img)

abs_threshold = gradient.AbsoluteThreshold((20,100))

mag_threshold = gradient.MagnitudeThreshold((30, 100))

dir_threshold = gradient.DirectionThreshold((0.7, 1.3))

combined = np.zeros_like(dir_threshold)
combined[((abs_threshold[0] == 1) & (abs_threshold[1] == 1)) | ((mag_threshold == 1) & (dir_threshold == 1))] = 1

f, (ax1, ax2) = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()
ax1[0].imshow(abs_threshold[1], cmap='gray')
ax1[0].set_title('Abs. Gra. Thr.', fontsize=50)
ax1[1].imshow(mag_threshold, cmap='gray')
ax1[1].set_title('Mag. Gra. Thr.', fontsize=50)
ax2[0].imshow(dir_threshold, cmap='gray')
ax2[0].set_title('Dir. Gra. Thr.', fontsize=50)
ax2[1].imshow(combined, cmap='gray')
ax2[1].set_title('Combined', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
