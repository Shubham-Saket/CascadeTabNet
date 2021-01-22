import os
import glob
import pandas as pd
import re
import xml.etree.ElementTree as ET

MODELS_CONFIG = {
    'faster_rcnn_r50_fpn_1x': {
        'config_file': 'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py'
    },
    'cascade_rcnn_r50_fpn_1x': {
        'config_file': 'configs/_base_/models/cascade_mask_rcnn_r50_fpn.py',
    },
    'retinanet_r50_fpn_1x': {
        'config_file': 'configs/retinanet_r50_fpn_1x.py',
    }
}

mmdetection_dir = "mmdetection"
project_name = "."
anno_path = os.path.join(project_name, "data/VOC2007/Annotations")
voc_file = os.path.join(mmdetection_dir, "mmdet/datasets/voc.py")

classes_names = []
xml_list = []
for xml_file in glob.glob(anno_path + "/*.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall("object"):
        classes_names.append(member[0].text)
classes_names = list(set(classes_names))
classes_names.sort()

fname = voc_file
with open(fname) as f:
    s = f.read()
    s = re.sub('CLASSES = \(.*?\)',
               'CLASSES = ({})'.format(", ".join(["\'{}\'".format(name) for name in classes_names])), s, flags=re.S)
with open(fname, 'w') as d:
    d.write(s)

