import cv2
# video_name = '../data/swedish2/video.avi'
# video_name = '/Users/smaket/Downloads/dvs-video-2.avi'
def play_video(video_name):
    vid_capture = cv2.VideoCapture(video_name)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    frame_idx = 0
    cv2.namedWindow('out')
    labels = []
    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret:
            cv2.imshow('out', frame)
            cv2.waitKey(50)
        else:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    play_video('/Users/smaket/PycharmProjects/DVS/v2e/test_video2.avi')