import glob
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
# freq_dir = '/Volumes/karakan/YOLO_filter/eventFreq/valid/obj/'
# frame_dir = '/Volumes/karakan/YOLO_filter/eventFrame/valid/obj/'
# dir_dir = '/Volumes/karakan/YOLO_filter/Merged/valid/obj'
# expTdecay_images = glob.glob('/Volumes/karakan/YOLO_filter/expTdecay/valid/obj/*.jpg')
# expTdecay_labels = glob.glob('/Volumes/karakan/YOLO_filter/expTdecay/valid/obj/*.txt')
#
# cv2.namedWindow('out')
# for image_path, label_path in zip(expTdecay_images, expTdecay_labels):
#     img_name = image_path.split('/')[-1]
#
#     print(img_name)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     print(image)
#     W, H = image.shape
#     new_image = np.zeros((W, H, 3))
#     new_image[:, :, 0] = image
#     try:
#         image_freq = cv2.imread(os.path.join(freq_dir, img_name), cv2.IMREAD_GRAYSCALE)
#         new_image[:, :, 1] = image_freq
#         image_frame = cv2.imread(os.path.join(frame_dir, img_name), cv2.IMREAD_GRAYSCALE)
#         new_image[:, :, 2] = image_frame
#         cv2.imshow('out', new_image)
#         cv2.waitKey(20)
#         cv2.imwrite(os.path.join(dir_dir, img_name), new_image)
#         label_name = img_name.replace('.jpg', '.txt')
#         shutil.copy(label_path, os.path.join(dir_dir, label_name))
#     except Exception as e:
#         pass

with open('/Volumes/karakan/YOLO_filter/Merged/train/train.txt', 'w') as f:
    lines = [f"data/Merged/train/obj/{jpg.split('/')[-1]}\n" for jpg in glob.glob('/Volumes/karakan/YOLO_filter/Merged/train/obj/*.jpg')]
    f.writelines(lines)