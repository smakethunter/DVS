import os
import re


def listobj(yolo_path):
    for set_part in os.listdir(yolo_path):
        print(set_part)
        if set_part in ['train', 'test', 'valid']:
            part = os.path.join(yolo_path, set_part)
            for part_agg_method in os.listdir(part):
                print(re.findall(r'DS_Store', part_agg_method))
                if len(re.findall(r'DS_Store', part_agg_method)) < 1:
                    obj_dir = os.path.join(part, part_agg_method, 'obj')
                    list_file = os.path.join(part, part_agg_method, f"{set_part}.txt")
                    print(obj_dir)
                    print(list_file)
                    with open(list_file, 'w') as f:
                        print(f"saving paths to {list_file}")
                        for filename in os.listdir(obj_dir):
                            if filename.split('.')[-1] == 'jpg':
                                # print('saving', f"{filename.split('.')[-1]} to {list_file}")
                                f.write(f"data/{set_part}/{part_agg_method}/obj/{filename.split('/')[-1]}\n")


# tests:
if __name__ == '__main__':
    listobj('/Volumes/karakan/YOLO')

