import os

import numpy as np
from dataset_utils.src.io.psee_loader import PSEELoader
from dataset_utils.src.visualize.vis_utils import make_binary_histo, draw_bboxes
from src.frame_aggregation.aggregators import maxdate_record

import cv2
def yolo2tlrb(dt, shape):
    dh, dw = shape
    x, y, w, h = tuple(dt)

    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
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
class FramesCreator:
    def __init__(self,
                 frame_size,
                 source_file_name,
                 filter_labels,
                 img_aggregation_type=make_binary_histo,
                 bboxes_aggregation_type=maxdate_record,
                 delta_t=1000,
                 thr=0
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
        img = self.img_aggregation_type(events, None)
        return img

    def add_boxes2YOLO(self, boxes):
        im_height, im_width = self._frame_size
        boxes = self.bboxes_aggregation_type(boxes)
        boxes_list = []
        for i in range(len(boxes)):
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
            (l, t), (r, b) = yolo2tlrb(box, image.shape[:2])
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

write_with_labels_dir = '/Volumes/karakan/YOLO_n_events/with_rect/'
write_obj_dir ='/Volumes/karakan/YOLO_n_events/obj/'
# dir='/Volumes/karakan/train/moorea_2019-01-30_000_td_305500000_365500000_td.dat'
dat_dir ='/Volumes/karakan/train2/'
cv2.namedWindow('out', cv2.WINDOW_NORMAL)

for file in [dat_dir+file+'_td.dat' for file in set(['_'.join(name.split('.')[0].split('_')[:-1]) for name in os.listdir(dat_dir)])]:
    dir = file
    video = PSEELoader(dir)
    npy_file = dir.replace('_td.dat', '_bbox.npy')
    boxes = np.load(npy_file)
    nr = 0
    filename = dir.split('/')[-1].split('.')[0]
    while not video.done:
        # load events and boxes from all files
        # bboxes = data = video_bb.load_n_events(10000)
        vid = video.load_n_events(30000)
        tmax, tmin = max(vid['t']), min(vid['t'])
        event_boxes = boxes[np.logical_and(boxes['t'] >= tmin, boxes['t'] <= tmax)]
        creator = FramesCreator((720, 1280),'name', filter_labels=[5])
        # im = make_binary_histo(events=vid, img=np.ones((720, 1280,3)), width=1280, height=720)
        im, bboxes = creator.add_frame(vid, event_boxes)
        try:

        # im_rgb = cv2.cvtColor(src=im, code=cv2.COLOR_GRAY2RGB)

            if len(boxes)>0:
                with open(f"{write_obj_dir}{filename}_{nr}.txt", 'w') as f:
                    f.writelines([' '.join(str(v) for v in box) + '\n' for box in bboxes])
                im_rgb = cv2.cvtColor(src=im, code=cv2.COLOR_GRAY2RGB)
                cv2.imwrite(f"{write_obj_dir}{filename}_{nr}.jpg", im_rgb)

                for box in bboxes:
                    vis_box = yolo2tlrb(box, im.shape[:2])
                    cv2.rectangle(im_rgb, pt1=vis_box[0], pt2=vis_box[1], color=(255,100,240))
                cv2.imwrite(f"{write_with_labels_dir}{filename}_{nr}.jpg", im_rgb)

                cv2.imshow('out', im_rgb)
                cv2.waitKey(1)
                nr += 1
        except Exception:
            pass
