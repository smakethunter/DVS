import json
import cv2

from utils.display_detections import blit_image
class_names = dict(zip(range(4), ['prohibitory',
'danger',
'mandatory',
'other'
]))

def display_frames_with_detections_in_json(json_path, frames_list, video_name):
    detections = json.load(open(json_path,'r'))
    img = []
    cv2.namedWindow('out')
    for frame_detections, frame in zip(detections,frames_list):
        print(frame)
        print(frame_detections)
        frame = cv2.imread(frame)
        boxes_rows = [[detection['relative_coordinates']['center_x'],
                       detection['relative_coordinates']['center_y'],
                       detection['relative_coordinates']['width'],
                       detection['relative_coordinates']['height']] for detection in frame_detections['objects']]

        labels = [class_names[detection["class_id"]] for detection in frame_detections['objects']]
        frame = blit_image(frame, boxes_rows, labels=labels, read_colormap=cv2.IMREAD_COLOR)
        cv2.imshow('out', frame)
        cv2.waitKey(1)
        img.append(frame)

    height, width, layers = img[1].shape
    cv2.destroyAllWindows()
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video = cv2.VideoWriter(video_name, apiPreference=0, fourcc=fourcc, frameSize=(width, height), fps=21)
    cv2.namedWindow('out')
    for j in img:
        video.write(j)
        cv2.imshow('out', j)
        cv2.waitKey(20)
if __name__ == '__main__':
    json_path = '../data/v2e/results4.json'
    data_path = '../data/dvs_test/data/obj'
    video_name = 'test_video3.avi'
    frames_list = [f"{data_path}/video990frame{i}.jpg" for i in range(300)]
    display_frames_with_detections_in_json(json_path, frames_list, video_name)