import os
import shutil

import cv2
import matplotlib.pyplot as plt

import os.path
import threading
from glob import glob
from operator import itemgetter
from itertools import groupby
import re
import cv2

from dataset_utils.src.io.psee_loader import PSEELoader
import math
import numpy as np


def group_by(func):
    def wrapper(*args, **kwargs):
        groups = groupby(np.sort(args[0], order='track_id'), itemgetter('track_id'))
        return np.array([func(g) for k, g in groups])

    return wrapper


@group_by
def maxdate_record(agroup):
    an_array = np.array(list(agroup))
    i = np.argmax(an_array['t'])
    return an_array[i]


@group_by
def mindate_record(agroup):
    an_array = np.array(list(agroup))
    i = np.argmin(an_array['t'])
    return an_array[i]


@group_by
def avgdate_record(agroup):
    an_array = np.array(list(agroup))
    return an_array[len(an_array) // 2]


def boxes2YOLO(boxes, im_height, im_width, file_name=None, agg_type='last'):
    """
    .txt-file for each .jpg-image-file - in the same directory and with the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, for each object in new line: <object-class> <x> <y> <width> <height>
    Where:
    <object-class> - integer number of object from 0 to (classes-1)
    <x> <y> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
    for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
    atention: <x> <y> - are center of rectangle (are not top-left corner)
    :param boxes:
    :param im_height:
    :param im_width:
    :return: <object-class> <x> <y> <width> <height>
    """
    with open(file_name) as f:
        for i in range(boxes.shape[0]):
            pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
            size = (int(boxes['w'][i]), int(boxes['h'][i]))
            pt2 = (pt1[0] + size[0], pt1[1] + size[1])
            class_id = boxes['class_id'][i]
            center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            w, h = size[0] / im_width, size[1] / im_height
            obj = f'{class_id} {center[0]} {center[1]} {w} {h}\n'


def frequency_aggregation(events, image_shape, delta_t=None):
    image = np.zeros(image_shape)
    for event in events:
        image[event['y'], event['x']] += event['p']
        image = 255 / (1 + np.exp(-image / 2))
        image = (image - 127.5)
        image = image.astype(np.uint8)
        image[image > 0] += 127
        image = np.clip(image, 0, 255)

        return image


def exponantial_decay_aggregation(events, image_shape, delta_t):
    image = np.zeros(image_shape)
    for event in events:
        image[event['y'], event['x']] += np.exp(-np.abs(event['t'] - event['p']) / delta_t)
    return image


def make_binary_histo(events, size, delta_t=None):
    img = np.zeros((size[0], size[1]), dtype=np.uint8)
    for event in events:
        img[event['y'], event['x']] = 255 * event['p']
    # image = (img - 127.5)
    # image = image.astype(np.uint8)
    # image[image > 0] += 127
    # image = np.clip(image, 0, 255)
    return img


class FramesCreator:
    def __init__(self,
                 frame_size,
                 recording_length,
                 source_file_name,
                 filter_labels,
                 img_aggregation_type,
                 bboxes_aggregation_type,
                 delta_t,
                 thr
                 ):

        """ .txt-file for each .jpg-image-file - in the same directory and with the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, for each object in new line: <object-class> <x> <y> <width> <height>
                  Where:
                  <object-class> - integer number of object from 0 to (classes-1)
                  <x> <y> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
                  for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
                  atention: <x> <y> - are center of rectangle (are not top-left corner)
                  :param boxes:
                  :param im_height:
                  :param im_width:
                  :return: <object-class> <x> <y> <width> <height>
        """
        self._frame_size = frame_size
        self._recording_counts = 0
        self.img_aggregation_type = img_aggregation_type
        self.bboxes_aggregation_type = bboxes_aggregation_type
        self.YOLO_file_name_prefix = '_'.join(source_file_name.split('_')[:-4])
        self.delta_t = delta_t
        self.thr = thr
        if filter_labels:
            self._filter_labels = {f: i for i, f in enumerate(filter_labels)}
        else:
            self._filter_labels = {}

    def add_events(self, events):
        img = self.img_aggregation_type(events, self._frame_size, self.delta_t)
        return img

    def add_boxes2YOLO(self, boxes):
        im_height, im_width = self._frame_size
        boxes = self.bboxes_aggregation_type(boxes)
        boxes_list = []
        for i in range(boxes.shape[0]):
            class_id = boxes['class_id'][i]
            if class_id in self._filter_labels.keys():
                pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
                size = (int(boxes['w'][i]), int(boxes['h'][i]))
                pt2 = (pt1[0] + size[0], pt1[1] + size[1])
                center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
                w, h = size[0] / im_width, size[1] / im_height
                box_center_x = center[0] / im_width
                box_center_y = center[1] / im_height
                if (box_center_x - w // 2 > 0 and box_center_y - h // 2 > 0) and (
                        box_center_x + w // 2 < 1 and box_center_y + h // 2 < 1):
                    boxes_list.append([box_center_x, box_center_y, w, h])
        return boxes_list

    def verify_obj_visibility(self, image, boxes):
        object_in_box = []
        for box in boxes:
            (l, t), (r, b) = yolo2tlrb(box, image.shape)
            box_area = image[t:b, l:r]
            area_of_the_box_in_pixels = (r - l) * (b - t)
            information = (box_area > 0)
            information = information.astype('uint8')
            active_pixels = np.sum(information, 0)
            active_pixels = np.sum(active_pixels, 0)
            if active_pixels / area_of_the_box_in_pixels > self.thr:
                object_in_box.append(True)
            else:
                object_in_box.append(False)
        if any(object_in_box):
            boxes_with_objects = []
            for i, box in enumerate(boxes):
                if object_in_box[i]:
                    boxes_with_objects.append(box)
            return True, boxes_with_objects
        else:
            return False, None

    def add_frame(self, events, boxes):
        img = None
        bx = None

        if self.filter_labels(boxes):
            bx = self.add_boxes2YOLO(boxes)
            img = self.add_events(events)
            is_any_obj, boxes_with_object = self.verify_obj_visibility(img, bx)
            if is_any_obj:
                return img, boxes_with_object
            else:
                return (None, None)
        return (None, None)

    def filter_labels(self, boxes):
        for label in self._filter_labels:
            if label in list(boxes['class_id']):
                return True
        return False


class Aggregator:
    def aggregate(self, dataset_name, n_events=None, delta_t=None):
        pass


class PropheseeDataAggregator(Aggregator):
    def __init__(self, img_aggregation_type, bboxes_aggregation_type, thr, write_dir, save=False, save_withbox=False):
        self.loader = PSEELoader
        self._video = None
        self._bboxes = None
        self._frames = None
        self.img_aggregation_type = img_aggregation_type
        self.bboxes_aggregation_type = bboxes_aggregation_type
        self.thr = thr
        self.write_dir = write_dir
        self.save = save
        self.save_withbbox = save_withbox

    def aggregate(self, td_files, n_events=None, delta_t=None, max_frames=4, filter_labels=[5], show=True):
        self._videos = [self.loader(td_file) for td_file in td_files]
        print([glob(td_file.split('_td.dat')[0] + '*.npy')[0] for td_file in td_files])
        self._box_videos = [self.loader(glob(td_file.split('_td.dat')[0] + '*.npy')[0]) for td_file in td_files]
        height, width = self._videos[0].get_size()
        length = np.sum([v.total_time() / delta_t for v in self._videos])
        creator = FramesCreator(frame_size=(height, width),
                                recording_length=length,
                                filter_labels=filter_labels,
                                img_aggregation_type=self.img_aggregation_type,
                                source_file_name=td_files[0].split('/')[-1],
                                bboxes_aggregation_type=self.bboxes_aggregation_type,
                                delta_t=delta_t,
                                thr=self.thr)
        if show:
            cv2.namedWindow('out', cv2.WINDOW_NORMAL)
        recording_counts = 0
        while not sum([video.done for video in self._videos]):
            events = [video.load_n_events(delta_t) for video in self._videos]
            delta_time = [max(vid['t']) - min(vid['t']) for vid in events]
            box_events = [box_video.load_n_events(delta) for box_video, delta in zip(self._box_videos, delta_time)]
            for index, (evs, boxes) in enumerate(zip(events, box_events)):
                image, boxes = creator.add_frame(events, boxes)
                source_file_name = td_files[0].split('/')[-1]
                if image is not None:
                    file_name = self.write_dir
                    file_name += '/' + '_'.join(source_file_name.split('_')[:-4]) + f'_{recording_counts}'

                    if show:
                        frame = blit_image(image.astype(np.uint8), boxes)
                        if self.save_withbbox:
                            print('saving', file_name)
                            cv2.imwrite(file_name + '.jpg', frame)
                            recording_counts += 1

                        cv2.imshow('out', frame)
                        cv2.waitKey(1)

                    if self.save:
                        cv2.imwrite(file_name + '.jpg', image.astype(np.uint8))
                        with open(file_name + '.txt', 'w') as f:
                            for b in boxes:
                                f.write(f"{' '.join([str(param) for param in b])}\n")
                        recording_counts += 1

    def aggregate_events_count(self, td_file, n_events=None, delta_t=None, max_frames=4, filter_labels=[5], show=True):
        self._video = self.loader(td_file)
        self._box_video = np.load(glob(td_file.split('_td.dat')[0] + '*.npy')[0])
        height, width = self._video.get_size()
        length = np.sum([v.total_time() / delta_t for v in self._videos])
        creator = FramesCreator(frame_size=(height, width),
                                recording_length=length,
                                filter_labels=filter_labels,
                                img_aggregation_type=self.img_aggregation_type,
                                source_file_name=td_file.split('/')[-1],
                                bboxes_aggregation_type=self.bboxes_aggregation_type,
                                delta_t=delta_t,
                                thr=self.thr)
        while not self._video.done:
            events = self._video.load_n_events(delta_t)
            tmax, tmin = max(events['t']), min(events['t'])
            box_events = self._box_video[np.logical_and(self._box_video['t'] >= tmin, self._box_video['t'] <= tmax)]
            for index, (evs, boxes) in enumerate(zip(events, box_events)):
                creator.add_frame(events, boxes)
            if show:
                cv2.namedWindow('out', cv2.WINDOW_NORMAL)
            recording_counts = 0
            while not sum([video.done for video in self._videos]):
                events = [video.load_n_events(delta_t) for video in self._videos]
                delta_time = [max(vid['t']) - min(vid['t']) for vid in events]
                box_events = [box_video.load_n_events(delta) for box_video, delta in zip(self._box_videos, delta_time)]
                for index, (evs, boxes) in enumerate(zip(events, box_events)):
                    image, boxes = creator.add_frame(events, boxes)
                    source_file_name = td_file.split('/')[-1]
                    if image is not None:
                        file_name = self.write_dir
                        file_name += '/' + '_'.join(source_file_name.split('_')[:-4]) + f'_{recording_counts}'

                        if show:
                            frame = blit_image(image.astype(np.uint8), boxes)
                            if self.save_withbbox:
                                print('saving', file_name)
                                cv2.imwrite(file_name + '.jpg', frame)
                                recording_counts += 1

                            cv2.imshow('out', frame)
                            cv2.waitKey(1)

                        if self.save:
                            cv2.imwrite(file_name + '.jpg', image.astype(np.uint8))
                            with open(file_name + '.txt', 'w') as f:
                                for b in boxes:
                                    f.write(f"{' '.join([str(param) for param in b])}\n")
                            recording_counts += 1


def yolo2tlrb(dt, shape):
    dh, dw = shape
    x, y, w, h = tuple(dt)

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    return (l, t), (r, b)


def blit_image(img, boxes):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for dt in boxes:
        lt, rb = yolo2tlrb(dt, img.shape[:2])

        cv2.rectangle(img, lt, rb, (255, 100, 100), 2)
    return img


def convert_events_to_frames(events_path, destination_path, show=True, save=False):
    conf_dict = {'event_frame': {'path_suffix': 'eventFrame',
                                 'function': make_binary_histo},
                 # 'freq': {'path_suffix': 'frequencyAggregation',
                 #          'function': frequency_aggregation},
                 # 'exp': {'path_suffix': 'expTdecay',
                 #         'function': exponantial_decay_aggregation}
                 }
    for agg_type, params in conf_dict.items():
        print(agg_type, params)
        data_dir = events_path
        print(data_dir)
        obj_dir = os.path.join(destination_path, params['path_suffix'], 'obj')
        file_list = os.listdir(data_dir)
        dat_list = []
        for f in file_list:
            if f.split('.')[-1] == 'dat':
                dat_list.append(os.path.join(data_dir, f))
        print(dat_list)

        agg_test = PropheseeDataAggregator(img_aggregation_type=conf_dict[agg_type]['function'],
                                           bboxes_aggregation_type=maxdate_record,
                                           thr=0,
                                           write_dir=obj_dir,
                                           save=save
                                           )
        # try:
        for file in dat_list:
            print(f"saving {file} to {obj_dir}")
            agg_test.aggregate([file], delta_t=100000, show=show)
        # except Exception as e:
        #     print(e)
        #     pass


def listobj(yolo_path):
    for set_part in os.listdir(yolo_path):
        print(set_part)
        if set_part in ['train', 'test', 'valid']:
            part = os.path.join(yolo_path, set_part)
            for part_agg_method in os.listdir(part):
                print(re.findall(r'DS_Store', part_agg_method))
                if len(re.findall(r'DS_Store', part_agg_method)) < 1:
                    obj_dir = os.path.join(part, part_agg_method, 'obj')
                    list_file = os.path.join(part, part_agg_method, f"{set_part}.txt")
                    print(obj_dir)
                    print(list_file)
                    with open(list_file, 'w') as f:
                        print(f"saving paths to {list_file}")
                        for filename in os.listdir(obj_dir):
                            if filename.split('.')[-1] == 'jpg':
                                # print('saving', f"{filename.split('.')[-1]} to {list_file}")
                                f.write(f"data/{set_part}/{part_agg_method}/obj/{filename.split('/')[-1]}\n")


def tlrb2yolo(l, t, r, b, shape):
    dw, dh = shape
    w = (r - l) / dw
    h = (b - t) / dh
    x = (r + l) / (2 * dw)
    y = (t + b) / (2 * dh)
    return x, y, w, h


if __name__ == '__main__':
    l, t, r, b = 100, 400, 400, 520

    x, y, w, h = tlrb2yolo(l, t, r, b, (1280, 720))
    print(x, y, w, h)
    print(yolo2tlrb((x, y, w, h), (1280, 720)))

# tests:
# if __name__ == '__main__':
#     # external_drive = '/Volumes/karakan/'
#     # agg = PropheseeDataAggregator(
#     #                               img_aggregation_type=frequency_aggregation,
#     #                               bboxes_aggregation_type=maxdate_record, thr=0, write_dir='../../../data')
#     # creator = agg.aggregate_events_count('/Volumes/karakan/train2/moorea_2019-02-18_000_td_427500000_487500000_td.dat',
#     #                                      delta_t=10000)
#     #
#     # # convert_events_to_frames('/Volumes/karakan/val', '/Volumes/karakan/YOLO/tt', save=False, show=True)
#
#     listobj('/Users/smaket/PycharmProjects/DVS/yolo/darknet/data')
#     write_dir = '/Volumes/karakan/YOLO/rect_valid/expTdecay/'
#     traindir = '/Volumes/karakan/YOLO/valid/expTdecay/obj/'
#     lines = [traindir+img+'.jpg' for img in set(file.split('.')[0] for file in os.listdir(traindir))]
#     cv2.namedWindow('out', cv2.WINDOW_NORMAL)
#     for name in lines:
#         img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
#         print((name.split('/')[-1].split('.')[0]))
#         if len(name.split('/')[-1].split('.')[0])>0:
#             with open(name.split('.')[0]+'.txt', 'r') as f:
#                 boxes_lines = f.readlines()
#             boxes = [[float(v) for v in b.split(' ')] for b in boxes_lines]
#             frame = blit_image(img, boxes)
#             n_write = name.split('/')[-1]
#             # print(n_write)
#             cv2.imwrite(write_dir+n_write, frame)
#             print(write_dir+n_write)
#             cv2.imshow('out', frame)
#             txt_name = '/'.join(write_dir.split('/')[:-1]) + name.split('.')[0]+'.txt'
#             shutil.copy(name.split('.')[0]+'.txt', txt_name)
