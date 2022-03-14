import subprocess

def make_vid2frames_directory(path, video_number):
    bashCommand1 = f"mkdir {path}/video{video_number}"
    process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand2 = f"mkdir video{video_number}/frames"
    process = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
video_number = 1
make_vid2frames_directory('.',video_number)