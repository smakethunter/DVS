import os
import shutil
# dest_dir = '/Volumes/karakan/YOLO/rect_valid/eventFreq/'
# source_folder = '/Volumes/karakan/YOLO/valid/expTdecay/obj/'#'/Volumes/karakan/YOLO/train/expTdecay/obj/'
# destination_folder = '/Volumes/karakan/YOLO_filter/expTdecay/valid/obj/'
#
# for obj in os.listdir(dest_dir):
#     print(obj)
#     if obj != '.DS_Store':
#         try:
#             shutil.copy(source_folder+obj, destination_folder+obj)
#             txt_file = obj.split('.')[0] + '.txt'
#             shutil.copy(source_folder + txt_file, destination_folder + txt_file)
#         except Exception:
#             pass
#
# # if __name__ == '__main__':
# train_dir = '/Volumes/karakan/YOLO_filter/expTdecay/valid/obj/'
# files = {l.split('.')[0] for l in os.listdir(train_dir)}
# lines = [f"data/expTdecay/valid/obj/{f+'.jpg'}\n" for f in files]
# print(lines)
# with open('/Volumes/karakan/YOLO_filter/expTdecay/valid/valid.txt', 'w') as file:
#     for line in lines:
#         file.write(line)
# # import matplotlib.pyplot as plt
# # im = plt.imread('/content/darknet/' + line, cv2.IMREAD_GRAYSCALE)
# #
# # im = cv2.cvtColor(src=im, code=cv2.COLOR_GRAY2RGB)
# # im = im / 255.
# # cv2.imwrite(line, im)
#
with open('valid.txt','w') as f:
    for i in range(990):
        f.write(f"data/obj/video990frame{i}.jpg\n")

import glob
import json

import pandas as pd
#
# class_names = dict(zip(range(4), ['prohibitory',
#                                   'danger',
#                                   'mandatory',
#                                   'other'
#                                   ]))
# class_count = {i: 0 for i in range(4)}
# jsons = glob.glob('/Users/smaket/PycharmProjects/DVS/data/dvs_test/data_before/results/*.json')
# for results_file in jsons:
#     results = json.load(open(results_file))
#     for frame in results:
#         for object in frame['objects']:
#             class_count[object['class_id']] += 1
# json.dump(class_count, open('clas_count.json', 'w'))
# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.rc('xtick', labelsize=15)
#
# matplotlib.rc('ytick', labelsize=15)
# pd.DataFrame({'Typ znaku': ['Zakaz',
#                         'Zagrożenie',
#                         'Nakaz',
#                         'Inne'
#                         ], 'Liczba znaków': [12384, 4449, 4646, 6912]}).plot.barh(x='Typ znaku', y='Liczba znaków', figsize =(10.5,12))
# # plt.show()
# plt.savefig('balance.jpg')