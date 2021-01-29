import os
import cv2
import numpy as np
import glob
from shutil import copy
import xml.etree.ElementTree as ET

PATH_TO_ORIGIAL_IMAGES = "C:\\Users\\saket\\Desktop\\cascade\\CascadeTabNet\\data\\Lableimg\\Lableimg"
PATH_TO_DEST = "C:\\Users\\saket\\Desktop\\cascade\\CascadeTabNet\\data\\Lableimg\\Lableimg"
ANNOTATION_PATH = "C:\\Users\\saket\\Desktop\\cascade\\CascadeTabNet\\data\\Contract Lock_Lableimg_Table\Contract Lock_Lableimg_Table"
img_files = glob.glob(PATH_TO_ORIGIAL_IMAGES+"/*")
total = len(img_files)

f = open("C:\\Users\\saket\\Desktop\\cascade\\CascadeTabNet\\Data Preparation\\coco.txt",'a')
def basicTransform(img):
    _, mask = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_not(mask)
    return img

# 2x2 Static kernal
kernal = np.ones((2,2),np.uint8)

for count,i in enumerate(img_files):
    image_name = os.path.basename(i)
    print("Progress : ",count,"/",total)
    img = cv2.imread(i,0)
    _, mask = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
    dst = cv2.dilate(mask,kernal,iterations = 1)
    dst = cv2.bitwise_not(dst)
    cv2.imwrite(PATH_TO_DEST+"\\Dilation_"+image_name,dst)
    tree = ET.parse(os.path.join(ANNOTATION_PATH,image_name.replace(".jpg",".xml")))
    tree.write(os.path.join(ANNOTATION_PATH,"Dilation_"+image_name.replace(".jpg",".xml")))
    f.write("\nDilation_"+image_name.replace(".jpg",""))

for count, i in enumerate(img_files):
    image_name = os.path.basename(i)
    print("Progress : ", count, "/", total)
    img = cv2.imread(i)

    # Split the 3 channels into Blue,Green and Red
    b, g, r = cv2.split(img)

    # Apply Basic Transformation
    b = basicTransform(b)
    r = basicTransform(r)
    g = basicTransform(g)

    # Perform the distance transform algorithm
    b = cv2.distanceTransform(b, cv2.DIST_L2, 5)  # ELCUDIAN
    g = cv2.distanceTransform(g, cv2.DIST_L1, 5)  # LINEAR
    r = cv2.distanceTransform(r, cv2.DIST_C, 5)  # MAX

    # Normalize
    r = cv2.normalize(r, r, 0, 1.0, cv2.NORM_MINMAX)
    g = cv2.normalize(g, g, 0, 1.0, cv2.NORM_MINMAX)
    b = cv2.normalize(b, b, 0, 1.0, cv2.NORM_MINMAX)

    # Merge the channels
    dist = cv2.merge((b, g, r))
    dist = cv2.normalize(dist, dist, 0, 4.0, cv2.NORM_MINMAX)
    dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)

    # In order to save as jpg, or png, we need to handle the Data
    # format of image
    data = dist.astype(np.float64) / 4.0
    data = 1800 * data  # Now scale by 1800
    dist = data.astype(np.uint16)

    # Save to destination
    cv2.imwrite(PATH_TO_DEST + "\\Smudge_" + image_name, dist)
    tree = ET.parse(os.path.join(ANNOTATION_PATH, image_name.replace(".jpg", ".xml")))
    tree.write(os.path.join(ANNOTATION_PATH, "Smudge_" + image_name.replace(".jpg", ".xml")))
    f.write("\nSmudge_" + image_name.replace(".jpg", ""))
f.close()