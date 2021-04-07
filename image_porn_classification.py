# Image Porn Classification
# Using LBP and Color Histogram Feature
# use example: classification(image path) return 1 (if image porn classified) 0 else

import cv2
from skimage import feature
import numpy as np
import math

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 2),
			range=(0, self.numPoints + 1))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist

def extract(imagePath):
    img = cv2.imread(imagePath)

    #color histogram extraction
    hist_blue = cv2.calcHist([img], [0], None, [25], [0, 256])
    hist_blue = cv2.normalize(hist_blue, hist_blue)

    hist_green = cv2.calcHist([img], [1], None, [25], [0, 256])
    hist_green = cv2.normalize(hist_green, hist_green)

    hist_red = cv2.calcHist([img], [2], None, [25], [0, 256])
    hist_red = cv2.normalize(hist_red, hist_red)

    hist_gr = (hist_green + hist_red + hist_blue) / 3
    hist_gr = hist_gr.ravel().tolist()

    # LBP extraction
    desc = LocalBinaryPatterns(24, 8)  # LocalBinaryPattern  descriptor using a numPoints=24 and radius=8.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    hist = hist.ravel().tolist()

    hist_gr = hist_gr + hist

    return hist_gr

def classification(imagePath):
    v = extract(imagePath)
    result = 0
    p_black = math.exp(
        v[1 - 1] * 2.3602 + v[2 - 1] * 0.6668 + v[3 - 1] * 2.9112 + v[4 - 1] * 0.4899 + v[5 - 1] * 2.4949 + v[
            6 - 1] * 9.9264 + v[7 - 1] * -5.8007 + v[8 - 1] * 10.7211 + v[9 - 1] * 2.012 + v[10 - 1] * 6.8944 + v[
            11 - 1] * -0.2959 + v[12 - 1] * 11.7475 + v[13 - 1] * -0.9949 + v[14 - 1] * 4.9072 + v[15 - 1] * 1.0702 + v[
            16 - 1] * 4.3913 + v[17 - 1] * 0.5623 + v[18 - 1] * 6.2437 + v[19 - 1] * -3.0554 + v[20 - 1] * 9.2697 + v[
            21 - 1] * 2.2582 + v[22 - 1] * 0.5915 + v[23 - 1] * 4.8789 + v[24 - 1] * -3.498 + v[25 - 1] * 7.758 + v[
            26 - 1] * 3.7598 + v[27 - 1] * -315.038 + v[28 - 1] * 148.032 + v[29 - 1] * -338.676 + v[
            30 - 1] * 495.5496 + v[31 - 1] * -187.0315 + v[32 - 1] * 314.1626 + v[33 - 1] * -692.9472 + v[
            34 - 1] * 431.445 + v[35 - 1] * -49.8537 + v[36 - 1] * 148.2845 + v[37 - 1] * 10.0198 + v[
            38 - 1] * -15.0696 + v[39 - 1] * -54.0982 + v[40 - 1] * 282.3828 + v[41 - 1] * -289.412 + v[
            42 - 1] * -36.233 + v[43 - 1] * -44.3589 + v[44 - 1] * 202.2039 + v[45 - 1] * -289.3867 + v[
            46 - 1] * 622.6034 + v[47 - 1] * 150.5843 + v[48 - 1] * -341.9958 + v[49 - 1] * 191.9233 + v[
            50 - 1] * 7.3827 + 1 * -14.15
    )
    p_normal = math.exp(
        v[1 - 1] * 0.5308 + v[2 - 1] * -0.4472 + v[3 - 1] * -2.6096 + v[4 - 1] * 2.7163 + v[5 - 1] * -4.1838 + v[
            6 - 1] * 9.7065 + v[7 - 1] * -14.506 + v[8 - 1] * 13.3718 + v[9 - 1] * -4.1703 + v[10 - 1] * 6.5987 + v[
            11 - 1] * -1.517 + v[12 - 1] * 4.1833 + v[13 - 1] * -3.8981 + v[14 - 1] * -0.936 + v[15 - 1] * 4.8338 + v[
            16 - 1] * 2.8479 + v[17 - 1] * 2.6839 + v[18 - 1] * -0.0153 + v[19 - 1] * -9.0584 + v[20 - 1] * 11.236 + v[
            21 - 1] * 2.7613 + v[22 - 1] * -3.2126 + v[23 - 1] * 2.1243 + v[24 - 1] * -2.3581 + v[25 - 1] * 1.1701 + v[
            26 - 1] * -12.5369 + v[27 - 1] * -75.6957 + v[28 - 1] * 240.7065 + v[29 - 1] * -300.454 + v[
            30 - 1] * 498.1207 + v[31 - 1] * 81.2484 + v[32 - 1] * 94.5994 + v[33 - 1] * -454.1977 + v[
            34 - 1] * 467.9669 + v[35 - 1] * -16.7077 + v[36 - 1] * 74.0285 + v[37 - 1] * 2.3599 + v[38 - 1] * -6.5922 +
        v[39 - 1] * -34.2052 + v[40 - 1] * 124.0791 + v[41 - 1] * -58.8021 + v[42 - 1] * -277.1535 + v[
            43 - 1] * 13.128 + v[44 - 1] * 47.4422 + v[45 - 1] * 24.4974 + v[46 - 1] * 479.308 + v[47 - 1] * -5.8474 +
        v[48 - 1] * -46.3387 + v[49 - 1] * -35.2698 + v[50 - 1] * 3.3177 + 1 * -8.7424)
    p_tot = 1 + p_black + p_normal

    prob_black = p_black / p_tot
    prob_normal = p_normal / p_tot
    prob_non = 1 / p_tot

    if prob_black > prob_normal:
        if prob_black > prob_non:
            result = 1
    else:
        if prob_normal > prob_non:
            result = 1

    return result

print(classification("test_other/1.jpg"))
print(classification("test_other/2.jpg"))
print(classification("test_other/3.jpg"))
print(classification("test_other/4.jpg"))
print(classification("test_other/5.jpg"))
print(classification("test_other/6.jpg"))
print(classification("test_other/7.jpg"))
print(classification("test_other/8.jpg"))
print(classification("test_other/9.jpg"))
print(classification("test_other/10.jpg"))
