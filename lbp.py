import cv2
import csv
from matplotlib import pyplot as plt
import glob
from skimage import feature
import numpy as np
import time

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

start_time = time.time()
header_lbp = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21','v22','v23','v24','v25','class']

with open('dataset_lbp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_lbp)

for imagePath in glob.glob("non/*.png"):
    img = cv2.imread(imagePath)
    # LBP
    desc = LocalBinaryPatterns(24, 8) #LocalBinaryPattern  descriptor using a numPoints=24 and radius=8.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    hist = hist.ravel().tolist()
    hist.append(imagePath)

    with open('dataset_lbp.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(hist)

for imagePath in glob.glob("non/*.jpg"):
    img = cv2.imread(imagePath)
    # LBP
    desc = LocalBinaryPatterns(24, 8) #LocalBinaryPattern  descriptor using a numPoints=24 and radius=8.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    hist = hist.ravel().tolist()
    hist.append(imagePath)

    with open('dataset_lbp.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(hist)

print("--- %s seconds ---" % (time.time() - start_time))