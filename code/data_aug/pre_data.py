import xml.etree.ElementTree as et

import os
import shutil

from PIL import Image

from collections import Counter

from utils.utils import get_classes

image_dir='VOCdevkit/VOC2007/JPEGImages'
annot_dir='VOCdevkit/VOC2007/Annotations'

train_len=1400

classes_path='model_data/cls_classes.txt'
classes,_= get_classes(classes_path)

print('class: ',classes)

for xml in os.listdir(annot_dir):
    #print('xml: ',xml)
    name,suf=os.path.splitext(xml)
    #print('name: ',name,suf)
    if 'aug' in name or suf!='.xml':
        continue
    #if name!='31':
    #    continue
    xml_pth=os.path.join(annot_dir,xml)
    tree=et.parse(xml_pth)
    root=tree.getroot()

    cls_ct=Counter()
    for obj in root.iter('object'):
        cls=obj.find('name').text
        if cls in classes:
            cls_ct[cls]+=1

    if cls_ct['crazing']+cls_ct['pitted_surface']>0.5*sum(cls_ct.values()):
        image_pth=os.path.join(image_dir,name+'.jpg')
        image=Image.open(image_pth)
        aug_image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        aug_image_pth=os.path.join(image_dir,name+'_aug.jpg')
        aug_image.save(aug_image_pth)
        print('save: ',aug_image_pth)

        for obj in root.iter('object'):
            bbox=obj.find('bndbox')
            xmin=bbox.find('xmin')
            xmax=bbox.find('xmax')
            x1=int(xmin.text)
            x2=int(xmax.text)
            #y1=int(bbox.find('ymin').text)
            #y2=int(bbox.find('ymax').text)
            xmin.text=str(200-x1)
            xmax.text=str(200-x2)
        aug_xml_pth=os.path.join(annot_dir,name+'_aug.xml')
        tree.write(aug_xml_pth)
        #break

print(len(os.listdir(annot_dir)),len(os.listdir(image_dir)))