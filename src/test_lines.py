import matplotlib.pyplot as plt

from lines import Lines

lines = Lines()

img = plt.imread('warped-example.jpg')
left_polynomial, rw_left_polynomial, left_fit, right_polynomial, rw_right_polynomial, right_fit = lines.Process(img, True)

y_eval = img.shape[0]
left_curvature = lines.CalculateCurvature(rw_left_polynomial, y_eval)
right_curvature = lines.CalculateCurvature(rw_right_polynomial, y_eval)

print('Left curvature: ' + str(left_curvature))
print('Right curvature: ' + str(right_curvature))
