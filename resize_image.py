import os
import cv2
from xml.etree import ElementTree as et
from math import floor,ceil

def round(x):
    if x- floor(x)>0.5:
        return ceil(x)
    return floor(x)


new_image_size = (500,500)

image_dir = r"C:\Users\saket\Desktop\cascade\CascadeTabNet\data\Lableimg\Lableimg"
annotation_dir = r"C:\Users\saket\Desktop\cascade\CascadeTabNet\data\Contract Lock_Lableimg_Table\Contract Lock_Lableimg_Table"
resix = os.path.join(os.path.dirname(image_dir),'resized_images')
annix = os.path.join(os.path.dirname(annotation_dir),'resized_ann')
if not os.path.exists(resix):
    os.mkdir(resix)
if not os.path.exists(annix):
    os.mkdir(annix)



for images in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,images))
    ann = et.parse(os.path.join(annotation_dir,images.replace('.jpg','.xml')))
    img_height_ratio = new_image_size[0]/img.shape[0]
    img_width_ratio = new_image_size[1] / img.shape[1]
    #image resizing
    img = cv2.resize(img,new_image_size,interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(resix,images),img)
    #annotation resizeing
    for tag in ann.findall('size/width'):
        tag.text = str(round(img_width_ratio * int(tag.text)))
    for tag in ann.findall('size/height'):
        tag.text = str(round(img_height_ratio * int(tag.text)))
    for tag in ann.findall('object/bndbox/xmin'):
        tag.text = str(round(img_width_ratio * int(tag.text)))
    for tag in ann.findall('object/bndbox/xmax'):
        tag.text = str(round(img_width_ratio * int(tag.text)))
    for tag in ann.findall('object/bndbox/ymin'):
        tag.text = str(round(img_height_ratio * int(tag.text)))
    for tag in ann.findall('object/bndbox/ymax'):
        tag.text = str(round(img_height_ratio * int(tag.text)))
    ann.write(os.path.join(annix,images.replace('.jpg','.xml')))
