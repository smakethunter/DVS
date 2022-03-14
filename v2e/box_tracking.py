import os

import pandas as pd
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
# import the necessary packages
import numpy as np
import argparse
import time
import cv2

from utils.display_detections import yolo2tlrb


class CentroidTracker():
    def __init__(self, maxDisappeared = 1):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        try:
            del self.objects[objectID]
            del self.disappeared[objectID]
        except KeyError:
            pass


    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
                # otherwise, are are currently tracking objects so we need to
                # try to match the input centroids to existing object
                # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                if D.shape[0] >= D.shape[1]:
                    # loop over the unused row indexes
                    for row in unusedRows:
                        # grab the object ID for the corresponding row
                        # index and increment the disappeared counter
                        objectID = objectIDs[row]
                        try:
                            self.disappeared[objectID] += 1
                        except KeyError:
                            self.disappeared[objectID] = 0
                        # check to see if the number of consecutive
                        # frames the object has been marked "disappeared"
                        # for warrants deregistering the object
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)

                else:
                    for col in unusedCols:
                        self.register(inputCentroids[col])
        # return the set of trackable objects
        return self.objects


def label_video(filename, video_name=None, video_files_folder=None, out_file='test.csv', delimiter=','):
    df = pd.read_csv(filename, delimiter=delimiter)
    # detections = df[['llx','lly','lrx','lry','ulx','uly','urx','ury']].values()
    if video_name is not None:
        vid_capture = cv2.VideoCapture(video_name)
        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
        frame_idx = 0
        ct = CentroidTracker()
        cv2.namedWindow('out')
        labels = []
        while (vid_capture.isOpened()):
            ret, frame = vid_capture.read()
            if ret == True:
                det = df[df['frameNumber'] == frame_idx]
                detections = det[['ulx', 'uly', 'lrx', 'lry', 'signType']].values
                if len(detections) > 0:
                    rects = [det.astype(int) for det in detections[:, :-1]]
                    objects = ct.update(rects)
                    for (objectID, centroid), name in zip(objects.items(), detections[:, -1]):
                        # draw both the ID of the object and the centroid of the
                        # object on the output frame
                        text = "ID {} name {}".format(objectID, name)
                        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        labels.append(objectID)
                    for box in detections:
                        box = box[:-1].astype(int)
                        (startX, startY, endX, endY) = tuple(box)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 255, 0), 2)

                cv2.imshow('out', frame)
                cv2.waitKey(20)
                cv2.imwrite(f'../data/swedish/out/frame{frame_idx}.jpg', frame)
            else:
                fill = [None]*(len(df) - len(labels))
                labels.extend(fill)
                df['labels'] = labels
                print(df)
                df.to_csv(out_file)
                print('saved')
                break
            frame_idx += 1
        vid_capture.release()
    else:
        labels = []
        ct = CentroidTracker()
        cv2.namedWindow('out')
        for frame_nr in range(df['Frame ID'].max()):
            frame = cv2.imread(video_files_folder + f'/frame{frame_nr}.jpg')
            det = df[df['Frame ID'] == frame_nr]
            detections = det[['x_center', 'y_center', 'bb_width', 'bb_heigth', 'class_']].values
            if len(detections) > 0:
                try:
                    rects = []
                    for det in detections[:, :-1]:
                        ul, lr = yolo2tlrb(det, frame.shape[:2])
                        rects.append([ul[0], ul[1], lr[0], lr[1]])
                    print(rects)
                    objects = ct.update(rects)
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
                        print((startX, startY), (endX, endY))
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
    label_video('../data/v2e/data/video0/results_vid1.csv',
                video_name=None,
                video_files_folder='../data/v2e/data/video0/frames',
                out_file='../data/v2e/data/video0/results_labeled.csv')

