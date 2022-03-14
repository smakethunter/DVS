import numpy as np
from scipy.interpolate import griddata
fst = np.array([4, 4, 1, 3])
snd = np.array([1, 1, 3, 4])
def interpolate_values(first_value, nr_points, last_value):
    linfit = griddata([0, nr_points], np.array([first_value, last_value]), list(range(1, nr_points-1)))
    return [list(l) for l in linfit]
interp = interpolate_values(fst,10,snd)

import cv2

def video_to_frames(video_path, folder_for_frames):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(folder_for_frames+"/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
def interpolate_for_frames(slow_mo_factor, boxes_on_video):
    prev_nr_frames = len(boxes_on_video)
    frames_between = int(1/slow_mo_factor)
    print(frames_between)
    old_frame_indexes = [x for x in range(0, int(prev_nr_frames/slow_mo_factor), frames_between)]
    boxes_for_new_video = []
    for idx, (old_frame_index,box) in enumerate(zip(old_frame_indexes[:-1], boxes_on_video[:-1])):
        new_boxes = [box]
        new_boxes.extend(interpolate_values(box, frames_between, boxes_on_video[idx+1]))

        print(len(new_boxes))
        # new_last_box_index=int(old_frame_index/slow_mo_factor)
        # new_first_box_index=int(old_frame_indexes[idx-1]/slow_mo_factor)
        # interpolated_boxes_indexes = [i for i in range(new_first_box_index, new_last_box_index)]
        # boxes_for_new_video.extend([i]+v for i, v in zip(interpolated_boxes_indexes, new_boxes))
        boxes_for_new_video.extend(new_boxes)
    return boxes_for_new_video
boxes_on_video = [[1,2], [2,3], [3,4]]
slow_mo = 0.1
interpolated_boxes = interpolate_for_frames(slow_mo,boxes_on_video)
print(interpolated_boxes)

