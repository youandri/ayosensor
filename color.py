import cv2
import csv
from matplotlib import pyplot as plt
import glob
import time

start_time = time.time()
header_color = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21','v22','v23','v24','v25','class']

with open('dataset_color.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_color)

for imagePath in glob.glob("non/*.png"):
    # Color Histogram
    img = cv2.imread(imagePath)
    hist_blue = cv2.calcHist([img],[0],None,[25],[0,256])
    hist_blue = cv2.normalize(hist_blue,hist_blue)

    hist_green = cv2.calcHist([img],[1],None,[25],[0,256])
    hist_green = cv2.normalize(hist_green,hist_green)

    hist_red = cv2.calcHist([img],[2],None,[25],[0,256])
    hist_red = cv2.normalize(hist_red,hist_red)

    plt.plot(hist_blue,color = 'b')
    plt.plot(hist_green,color = 'g')
    plt.plot(hist_red,color = 'r')

    hist_gr = (hist_green+hist_red+hist_blue)/3
    hist_gr = hist_gr.ravel().tolist()
    hist_gr.append(imagePath)

    with open('dataset_color.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(hist_gr)

for imagePath in glob.glob("non/*.jpg"):
    # Color Histogram
    img = cv2.imread(imagePath)
    hist_blue = cv2.calcHist([img],[0],None,[25],[0,256])
    hist_blue = cv2.normalize(hist_blue,hist_blue)

    hist_green = cv2.calcHist([img],[1],None,[25],[0,256])
    hist_green = cv2.normalize(hist_green,hist_green)

    hist_red = cv2.calcHist([img],[2],None,[25],[0,256])
    hist_red = cv2.normalize(hist_red,hist_red)

    plt.plot(hist_blue,color = 'b')
    plt.plot(hist_green,color = 'g')
    plt.plot(hist_red,color = 'r')

    hist_gr = (hist_green+hist_red+hist_blue)/3
    hist_gr = hist_gr.ravel().tolist()
    hist_gr.append(imagePath)

    with open('dataset_color.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(hist_gr)

print("--- %s seconds ---" % (time.time() - start_time))