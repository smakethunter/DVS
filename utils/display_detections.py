import cv2
import numpy as np


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


def blit_image(img, boxes, labels=None, read_colormap=cv2.COLOR_GRAY2RGB):
    img = cv2.cvtColor(img, read_colormap)

    for i, dt in enumerate(boxes):
        try:
            print(dt)
            lt, rb = yolo2tlrb(dt, img.shape[:2])
            print(lt, rb)
            cv2.rectangle(img, lt, rb, (255, 100, 100), 10)
            if labels:
                cv2.putText(img, labels[i], (lt[0], lt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception:
            pass
    return img

if __name__ == '__main__':
    import pandas as pd
    video_name = '/Volumes/karakan/dvs_videos/video0/01_01_01_01_01.mp4'
    csv_path = '../data/v2e/data/video0/resultvideo0.csv'
    detections = pd.read_csv(csv_path)
    cv2.namedWindow('out')
    img = []
    # for frame_nr in range(detections['Frame ID'].max()):
    #     frame = cv2.imread(image_path+f'frame{frame_nr}.jpg')
    #     boxes_rows = detections[['x_center', 'y_center', 'bb_width', 'bb_heigth']][detections['Frame ID'] == frame_nr].to_numpy()
    #     boxes_rows = list(boxes_rows)
    #     if boxes_rows[0][0] is not np.nan:
    #         frame = blit_image(frame, boxes_rows, read_colormap=cv2.IMREAD_COLOR)
    #         cv2.imshow('out', frame)
    #         cv2.waitKey(500)
    #     else:
    #         cv2.imshow('out', frame)
    #         cv2.waitKey(500)
    vid_capture = cv2.VideoCapture(video_name)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    frame_idx = 0
    cv2.namedWindow('out')
    labels = []
    frame_nr = 0
    try:
        while (vid_capture.isOpened()):
            ret, frame = vid_capture.read()
            if ret:
                boxes_rows = detections[['x_center', 'y_center', 'bb_width', 'bb_heigth']][
                detections['Frame ID'] == frame_nr].to_numpy()
                boxes_rows = list(boxes_rows)
                if boxes_rows[0][0] is not np.nan:
                    frame = blit_image(frame, boxes_rows, read_colormap=cv2.IMREAD_COLOR)
                    cv2.imshow('out', frame)
                frame_nr += 1
                cv2.waitKey(1)
                img.append(frame)
            else:
                break
    except Exception as e:
        pass
    height, width, layers = img[1].shape
    cv2.destroyAllWindows()
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video=cv2.VideoWriter('test_video.avi', apiPreference=0,fourcc=fourcc, frameSize=(width, height), fps=21)
    cv2.namedWindow('out')
    for j in img:
        video.write(j)
        cv2.imshow('out', j)
        cv2.waitKey(20)
    video.release()



