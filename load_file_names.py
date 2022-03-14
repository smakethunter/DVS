import os

data_dir = 'data/yolov4/'
image_names_file = data_dir + 'train.txt'
obj_dir = data_dir + 'obj'
with open(image_names_file, 'w') as f:
    for filename in os.listdir(obj_dir):
        if filename.split('.')[-1] == 'jpg':
            f.write(f"data/obj/{filename.split('/')[-1]}\n")
        