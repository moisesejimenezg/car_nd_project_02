import matplotlib.pyplot as plt

from lines import Lines

lines = Lines()

img = plt.imread("../examples/warped-example.jpg")
left_fit, right_fit = lines.Process(img, True)

y_eval = img.shape[0]
left_curvature = lines.CalculateCurvature(left_fit.rw_polynomial_, y_eval)
right_curvature = lines.CalculateCurvature(right_fit.rw_polynomial_, y_eval)

print("Left curvature: " + str(left_curvature))
print("Right curvature: " + str(right_curvature))
