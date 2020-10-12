import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from advanced_lane_finder.core.calibration import Calibration

calibration = Calibration(9, 6)

# Make a list of calibration images
files = glob.glob("advanced_lane_finder/data/camera_cal/calibration*.jpg")

for image_name in files:
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    calibration.Update(gray)

calibration.Calibrate()
image_name = "advanced_lane_finder/data/camera_cal/calibration1.jpg"
img = cv2.imread(image_name)
undistorted = calibration.Undistort(img)
plt.imshow(undistorted)
plt.show()
