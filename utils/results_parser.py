import json
import os


def results_json_to_txt_files(json_file, frames_path):
    with open(json_file, 'r') as f:
        detections = json.load(f)
    videoid = json_file.split('/')[-1].split('.')[0].replace('result', '')
    print(videoid)
    for idx, frame in enumerate(detections):
        with open(os.path.join(frames_path, f'{videoid}frame{idx}.txt'), 'w') as f:
            for single_detection in frame['objects']:
                center_x = single_detection['relative_coordinates']['center_x']
                center_y = single_detection['relative_coordinates']['center_y']
                bb_width = single_detection['relative_coordinates']['width']
                bb_height = single_detection['relative_coordinates']['height']
                confidence = single_detection['confidence']
                class_ = single_detection['class_id']
                f.write(f'{class_} {center_x} {center_y} {bb_width} {bb_height}\n')

if __name__ == '__main__':
    results_dir = '../data/dvs_test/data/result2'
    dest_dir = '../data/dvs_test/data_processed/obj'
    for json_file in sorted(os.listdir(results_dir)):
        json_file = os.path.join(results_dir, json_file)
        results_json_to_txt_files(json_file, dest_dir)

