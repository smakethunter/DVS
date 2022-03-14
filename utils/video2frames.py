import csv
import glob
import json
import os
import shutil

import cv2
import subprocess

import pandas as pd

from display_detections import yolo2tlrb

def make_vid2frames_directory(path, video_number):
    bashCommand1 = f"mkdir {path}/video{video_number}"
    process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand2 = f"mkdir {path}/video{video_number}/frames"
    process = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def videos2frames(data_path, video_idx, video_name):
        save_dir = f'{data_path}/video{video_idx}/frames'
        file_txt = f'{data_path}/video{video_idx}/train.txt'
        data_file = f'{data_path}/video{video_idx}/obj.data'
        vid_capture = cv2.VideoCapture(video_name)
        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
        frame_idx = 0
        with open(file_txt, 'w') as f:
            while (vid_capture.isOpened()):
                ret, frame = vid_capture.read()
                if ret:
                    cv2.imwrite(save_dir + f'/video{video_idx}frame{frame_idx}.jpg', frame)
                    f.write(os.path.join(f'data/video{video_idx}/frames', f'video{video_idx}frame{frame_idx}.jpg\n'))
                    frame_idx += 1
                else:
                    break
                if cv2.waitKey(1) | 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
            vid_capture.release()
        with open(data_file, 'w') as f:
            f.write(f'valid = data/video{video_idx}/valid.txt\n')
            f.write('names = data/classes.names\n')

def json2csv(json_path, csv_path):
    with open(json_path, encoding='latin-1') as json_file:
        data = json.load(json_file)

    # open csv file
    csv_file_to_make = open(csv_path, 'w', newline='\n')

    csv_file = csv.writer(csv_file_to_make)

    # write the header
    # NB x and y values are relative
    csv_file.writerow(['Frame ID',
                       'class_',
                       'x_center',
                       'y_center',
                       'bb_width',
                       'bb_heigth',
                       'confidence'])

    for frame in data:
        frame_id = frame['frame_id'] - 1
        instrument = ""
        center_x = ""
        center_y = ""
        bb_width = ""
        bb_height = ""
        confidence = ""
        class_ = ""

        if frame['objects'] == []:
            csv_file.writerow([frame_id,
                               class_,
                               center_x,
                               center_y,
                               bb_width,
                               bb_height,
                               confidence
                               ])
        else:
            for single_detection in frame['objects']:
                instrument = single_detection['name']
                center_x = single_detection['relative_coordinates']['center_x']
                center_y = single_detection['relative_coordinates']['center_y']
                bb_width = single_detection['relative_coordinates']['width']
                bb_height = single_detection['relative_coordinates']['height']
                confidence = single_detection['confidence']
                class_ = single_detection['class_id']
                csv_file.writerow([frame_id,
                                   class_,
                                   center_x,
                                   center_y,
                                   bb_width,
                                   bb_height,
                                   confidence
                                   ])

    csv_file_to_make.close()
    # cmd = f"rm {json_path}"
    # process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()

def jsons_to_csvs(videos_dir='data'):
    for video_dir in sorted(os.listdir(videos_dir)):
        json2csv(f'{videos_dir}/{video_dir}/result.json', f'{video_dir}/{video_dir}/result.csv')
        bashCommand1 = f"rm {videos_dir}/{video_dir}/results.json"
        process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
def csv_yolo2rlxy(csv_path, csv_path_write, image_shape):
    csv_yolo = pd.read_csv(csv_path)

    csv_file_to_make = open(csv_path_write, 'w', newline='\n')

    csv_file = csv.writer(csv_file_to_make)
    csv_file.writerow(['frameNumber',
                       'category',
                       'ulx',
                       'uly',
                       'lrx',
                       'lry',
                       ])

    for i in range(len(csv_yolo)):
        row = csv_yolo.iloc[i]
        try:
            (l, t), (r, b) = yolo2tlrb(row[['x_center', 'y_center', 'bb_width', 'bb_heigth']], shape=image_shape)
            csv_file.writerow([int(row['Frame ID']),
                               int(row['class_']),
                               l,
                               t,
                               r,
                               b])
        except Exception as e:
            print(e)
            pass


    csv_file_to_make.close()

def detections_on_videos(data_dir, videos_dir, cfg_dir, darknet_dir='darknet'):
    bashCommand1 = f"rm -r data"
    process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)
    bashCommand1 = f"mkdir data"
    process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)
    videos2frames(data_dir, videos_dir)

def videos2frames_for_training(data_path, video_idx, video_name):
    save_dir = f'{data_path}/obj'
    file_txt = f'{data_path}/train.txt'
    vid_capture = cv2.VideoCapture(video_name)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    frame_idx = 0
    with open(file_txt, 'a') as f:
        while (vid_capture.isOpened()):
            ret, frame = vid_capture.read()
            if ret:
                cv2.imwrite(save_dir + f'/video{video_idx}frame{frame_idx}.jpg', frame)
                f.write(os.path.join(f'data/obj', f'video{video_idx}frame{frame_idx}.jpg\n'))
                frame_idx += 1
            else:
                break
            if cv2.waitKey(1) | 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        vid_capture.release()

def save_videos_for_training(videos_dir, data_path):
    for idx, video_folder in enumerate(sorted(os.listdir(videos_dir))):
        video_name = os.path.join(videos_dir,
                                  video_folder,
                                  os.listdir(os.path.join(videos_dir, video_folder))[0])

        i = int(video_folder.replace('video', ''))
        videos2frames_for_training(data_path, i, video_name)


if __name__ == '__main__':
    videos2frames('../results', 0, '../v2e/test_video.avi')
    # save_videos_for_training('../data/dvs_test/data/output_dvs', '../data/dvs_test/data_processed')
    # detections_on_videos('../data/v2e/data', '../data/v2e/videos', 'cfg')
    # csv_yolo2rlxy('../data/v2e/data/video0/results.csv', '../data/v2e/data/video0/results2.csv', (1628, 1236))
    # videos = glob.glob('/Volumes/karakan/CURE-TSD/data/01_[0-9][0-9]_01_[0-9][0-9]_0[0-2].mp4')
    # path = '/Volumes/karakan/data_videos'
    # for video_number, video in enumerate(videos):
    #     shutil.move(video, f"{path}/video{video_number}/{video.split('/')[-1]}")

    # path = '../data/v2e/data/video0/dvs-video.avi'
    # #     videos2frames(path, video_number, video)
    # data_path = '../data/dvs/data'
    # make_vid2frames_directory(data_path, 0)
    # videos2frames(data_path, 0, path)
    # json_path = '/Users/smaket/PycharmProjects/DVS/data/dvs_test/data/results/resultvideo0.json'
    # csv_path = '../data/v2e/data/video0/resultvideo0.csv'
    # json2csv(json_path, csv_path)
