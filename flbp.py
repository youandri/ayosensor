import cv2
import math
import time
import csv
import glob
import numpy as np

def flbp(matrixCitra, neighbors, radius):
    blok = 2*radius + 1
    F = 4
    flbpWidth = 1 + len(matrixCitra[1]) - blok
    flbpHeight = 1 + len(matrixCitra) - blok

    histClbp = [0.00] * int(math.pow(2, neighbors))

    for y in range(0, flbpHeight):
        for x in range(0, flbpWidth):
            white = 0
            for i in range(0, blok):
                for j in range(0, blok):
                    #check white pixel in local region
                    if matrixCitra[y + i][x + j]==255:
                        white = white + 1

            #hitung sudut antarpixel tetangga
            angle = 2*math.pi/neighbors

            #posisi pixel pusat
            centerX = int(x) + int(blok / 2)
            centerY = int(y) + int(blok / 2)

            #menampung kodel lbp
            arl = [0]

            #menampung nilai clbp
            clbp = [1.00]

            for i in range(0, neighbors):
                posX = int(centerX)+ int(round((radius + 0.1) * math.cos(i * angle)))
                posY = int(centerY) - int(round((radius + 0.1) * math.sin(i * angle)))

                #mencari delta Pi int
                deltaP = int(matrixCitra[posY][posX]) - int(matrixCitra[centerY][centerX])

                #fuzzy thresholding
                if deltaP >= F:
                    for p in range(0, len(arl)):
                        temp = int(arl[p])
                        temp = temp + int(math.pow(2, i))
                        arl[p] = temp

                if deltaP > -1 * F and deltaP < F:
                    #buat cabang baru
                    jum_arl = len(arl)
                    for p in range(0, jum_arl):
                        arl = arl + [0]
                        clbp = clbp + [1.00]

                    #hitung kode lbp
                    q = 0
                    for p in range(jum_arl, len(arl)):
                        temp = int(arl[q])
                        temp = temp + int(math.pow(2, i))
                        arl[p] = temp
                        q = q + 1

                    #hitung clbp m0
                    median = int(len(clbp)/2)
                    for r in range(0, median):
                        mf = float(clbp[r])
                        mf = mf * float((F - deltaP)/(2*F))
                        clbp[r] = mf

                    #hitung clbp m1
                    a = 0
                    for s in range(median, len(clbp)):
                        mf = float(clbp[a])
                        mf = mf * float((F + deltaP)/(2*F))
                        clbp[s] = mf
                        a = a + 1

            for i in range(0, len(arl)):
                lbpVal = int(arl[i])
                clbpval = float(clbp[i])
                histclbp = float(histClbp[lbpVal])
                histclbp += clbpval
                if white != blok*blok:
                    histClbp[lbpVal] = histclbp
    return histClbp

start_time = time.time()
header_flbp = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21','v22','v23','v24','v25','v26','v27','v28','v29','v30','v31','v32','v33','v34','v35','v36','v37','v38','v39','v40','v41','v42','v43','v44','v45','v46','v47','v48','v49','v50','v51','v52','v53','v54','v55','v56','v57','v58','v59','v60','v61','v62','v63','v64','v65','v66','v67','v68','v69','v70','v71','v72','v73','v74','v75','v76','v77','v78','v79','v80','v81','v82','v83','v84','v85','v86','v87','v88','v89','v90','v91','v92','v93','v94','v95','v96','v97','v98','v99','v100','v101','v102','v103','v104','v105','v106','v107','v108','v109','v110','v111','v112','v113','v114','v115','v116','v117','v118','v119','v120','v121','v122','v123','v124','v125','v126','v127','v128','v129','v130','v131','v132','v133','v134','v135','v136','v137','v138','v139','v140','v141','v142','v143','v144','v145','v146','v147','v148','v149','v150','v151','v152','v153','v154','v155','v156','v157','v158','v159','v160','v161','v162','v163','v164','v165','v166','v167','v168','v169','v170','v171','v172','v173','v174','v175','v176','v177','v178','v179','v180','v181','v182','v183','v184','v185','v186','v187','v188','v189','v190','v191','v192','v193','v194','v195','v196','v197','v198','v199','v200','v201','v202','v203','v204','v205','v206','v207','v208','v209','v210','v211','v212','v213','v214','v215','v216','v217','v218','v219','v220','v221','v222','v223','v224','v225','v226','v227','v228','v229','v230','v231','v232','v233','v234','v235','v236','v237','v238','v239','v240','v241','v242','v243','v244','v245','v246','v247','v248','v249','v250','v251','v252','v253','v254','v255','v256','class']

with open('dataset_flbp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_flbp)

for imagePath in glob.glob("black/*.jpg"):
    img = cv2.imread(imagePath)
    matrixCitra = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrixCitra = cv2.resize(matrixCitra, (50, 100))
    hist = flbp(matrixCitra, 8, 3)
    hist = np.array(hist).tolist()
    hist.append(imagePath)

    with open('dataset_flbp.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(hist)

for imagePath in glob.glob("normal/*.jpg"):
    img = cv2.imread(imagePath)
    matrixCitra = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrixCitra = cv2.resize(matrixCitra, (50, 100))
    hist = flbp(matrixCitra, 8, 3)
    hist = np.array(hist).tolist()
    hist.append(imagePath)

    with open('dataset_flbp.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(hist)

print("--- %s seconds ---" % (time.time() - start_time))