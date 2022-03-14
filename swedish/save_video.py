import os

import cv2
import numpy as np
import pandas as pd
print(sorted(os.listdir('/Users/smaket/PycharmProjects/DVS/data/swedish2/Set2Part0')))
sequence = 1277104458
df = pd.read_csv('../data/swedish2/annotations.csv')
df['sequence'] = df['image'].apply(lambda x: int(x.split('Image')[0]))
sequence_for_video = df[df['sequence']==sequence]

img=[]
for frame in np.unique(sequence_for_video['image']):
    img.append(cv2.imread('../data/swedish2/Set2Part0/'+frame))
    print(img[-1].shape, frame)
height, width, layers = img[1].shape
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video=cv2.VideoWriter('../data/swedish2/video.avi', apiPreference=0,fourcc=fourcc, frameSize=(width, height), fps=9)
cv2.namedWindow('out')
for j in img:
    video.write(j)
    cv2.imshow('out', j)
    cv2.waitKey(20)
video.release()