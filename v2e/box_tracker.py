import pandas as pd
import numpy as np
import cv2
from utils.display_detections import yolo2tlrb
class BBox:
    def __init__(self, bbox, image_shape):

        (H, W) = image_shape
        (self.x1,self.y1), (self.x2, self.y2) = yolo2tlrb(bbox[:3], image_shape)
        self.x_center = bbox[0] * W
        self.y_center = bbox[1] * H
        self.category = bbox[4]

    
class ObjectTracker:
    def __init__(self, window_size):
        self.present_objects = {}
        self.time_of_absence = {}
        self.is_absent_in_frame={}
        self.next_free_id = 0
        self.window_size = window_size

    @staticmethod
    def distance(bbox1: BBox, bbox2: BBox):
        return np.sqrt((bbox1.x_center - bbox2.x_center) ** 2 + (bbox1.y_center - bbox2.y_center) ** 2)

    @staticmethod
    def get_iou(bbox1: BBox, bbox2: BBox):
        if bbox1.x1 < bbox1.x2 and bbox1.y1 < bbox1.y2 and bbox2.x1 < bbox2.x2 and bbox2.x1 < bbox2.x2 and bbox2.y1 < bbox2.y2:
            x_left = max(bbox1.x1, bbox2.x1)
            y_top = max(bbox1.y1, bbox2.y1)
            x_right = min(bbox1.x2, bbox2.x2)
            y_bottom = min(bbox1.y2, bbox2.y2)
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            bbox1_area = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
            bbox2_area = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
            iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
            if 0.0 <= iou <= 1.0:
                return iou

        return 0
    
    def register(self, bbox):
        self.present_objects[self.next_free_id] = bbox
        self.time_of_absence[self.next_free_id] = 0
        self.is_absent_in_frame[self.next_free_id] = False
        self.next_free_id += 1

    def deregister(self, id):
        del self.present_objects[id]
        del self.is_absent_in_frame[id]
        del self.time_of_absence[id]


    def process_bbox(self, bbox, iou_thr=0):

        if self.present_objects:
            min_distance = np.inf
            closest_bbox_id = None

            for existing_box_id, existing_bbox in self.present_objects.items():
                distance = self.distance(existing_bbox, bbox)
                if distance < min_distance:
                    min_distance = distance
                    closest_bbox_id = existing_box_id
            # check if closest centroids has the intersect
            if self.get_iou(bbox, self.present_objects[closest_bbox_id]) > iou_thr:
                # mark closest bbox as existing
                self.is_absent_in_frame[closest_bbox_id] = False

                pass
            else:
                self.register(bbox)

        else:
            self.register(bbox)

    def presence_reload(self):
        for k, v in self.is_absent_in_frame.items():
            self.is_absent_in_frame[k] = True

    def presence_update(self):
        for k, v in self.is_absent_in_frame.items():
            if self.is_absent_in_frame[k]:
                self.time_of_absence[k] += 1
        for k, v in self.time_of_absence.items():
            if self.time_of_absence >= self.window_size:
                self.deregister(k)

    def process_frame(self, bboxes, iou_thr=0):
        self.presence_reload()
        for bbox in bboxes:
            self.process_bbox(bbox, iou_thr)
        self.presence_update()
        return self.present_objects

    def get_labeled_boxes(self, bboxes, image_shape, iou_thr=0):
        bboxes = (BBox(bbox, image_shape) for bbox in bboxes)
        return self.process_frame(bboxes, iou_thr)

def track_objects(filename, video_files_folder, out_file='test.csv', delimiter=','):
    df = pd.read_csv(filename, delimiter=delimiter)
    labels = []
    ot = ObjectTracker(window_size=1)
    cv2.namedWindow('out')
    for frame_nr in range(df['Frame ID'].max()):
        frame = cv2.imread(video_files_folder + f'/frame{frame_nr}.jpg')
        det = df[df['Frame ID'] == frame_nr]
        detections = det[['x_center', 'y_center', 'bb_width', 'bb_heigth', 'class_']].values
        if len(detections) > 0:
            try:
                rects = [det for det in detections]
                print(rects)
                objects = ot.get_labeled_boxes(rects, frame.shape[:2])
                for (objectID, centroid), name in zip(objects.items(), detections[:, -1]):
                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {} name {}".format(objectID, name)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    labels.append(objectID)
            except Exception:
                labels.append(None)
                pass
            for box in detections:
                try:
                    box = box[:-1]
                    (startX, startY), (endX, endY) = yolo2tlrb(box, frame.shape[:2])
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                except Exception:
                    pass

        cv2.imshow('out', frame)
        cv2.waitKey(100)
        # cv2.imwrite(f'../data/swedish/out/frame{1}.jpg', frame)

    fill = [None] * (len(df) - len(labels))
    labels.extend(fill)
    df['labels'] = labels
    print(df)
    df.to_csv(out_file)
    print('saved')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    filename = '../data/swedish/annotations.csv'
    video_name = '../data/swedish/movie.mov'
    track_objects('../data/v2e/data/video0/results_vid1.csv',
                video_files_folder='../data/v2e/data/video0/frames',
                out_file='../data/v2e/data/video0/results_labeled.csv')
