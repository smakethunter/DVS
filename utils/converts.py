import os
import cv2
from glob import glob
def images_from_dir_to_gray(dir):
    filenames = glob(f'{dir}/*.jpg')
    for file in filenames:
        img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(file, img)

def make_video_from_frames(frames_dir, video_dir, start, end, fps=30):
    img = []
    for frame in range(start,end):
        img.append(cv2.imread(os.path.join(frames_dir, f'frame{frame}.jpg')))
    height, width, layers = img[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_dir, apiPreference=0, fourcc=fourcc, frameSize=(width, height),
                            fps=fps)
    cv2.namedWindow('out')
    for j in img:
        video.write(j)
        cv2.imshow('out', j)
        cv2.waitKey(20)
    video.release()
if __name__ == '__main__':
    # images_from_dir_to_gray('/Volumes/karakan/data/obj')
    # dir = '/Volumes/karakan/data/obj'
    # filenames = glob(f'{dir}/*.jpg')
    # print(cv2.imread(filenames[0]).shape)
    make_video_from_frames('../data/v2e/data/video0/frames', 'video.mp4', 120, 150, 21)
    from swedish.play_video import play_video
    play_video('video.mp4')
