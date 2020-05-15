import cv2
import matplotlib.pyplot as plt

from pipeline import Pipeline

pipeline = Pipeline(9, 6)
pipeline.Calibrate('../camera_cal/calibration*.jpg')
img = plt.imread('../test_images/test1.jpg')
pipeline.Undistort(img)
pipeline.CalculateGradient()
pipeline.FilterGradients((20,100), (30, 100), (0.7, 1.3))
pipeline.InitColor()
pipeline.FilterColor((170, 255))

combinedA = pipeline.JoinOption('A')
combinedB = pipeline.JoinOption('B')
combinedC = pipeline.JoinOption('C')

f, (ax1, ax2) = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()
ax1[0].imshow(img)
ax1[0].set_title('Raw', fontsize=10)
ax1[1].imshow(combinedA, cmap='gray')
ax1[1].set_title('Option A', fontsize=10)
ax2[0].imshow(combinedB, cmap='gray')
ax2[0].set_title('Option B', fontsize=10)
ax2[1].imshow(combinedC, cmap='gray')
ax2[1].set_title('Option C', fontsize=10)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
