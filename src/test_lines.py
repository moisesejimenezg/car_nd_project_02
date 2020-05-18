import matplotlib.pyplot as plt

from lines import Lines

lines = Lines()

img = plt.imread('warped-example.jpg')
histogram = lines.Process(img, True)
