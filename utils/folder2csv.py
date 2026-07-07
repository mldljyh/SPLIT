import os
import pandas as pd
from pandas import Series
from glob import glob
import os

def main(is_real,dataset_path,folder_paths):

    def count_images_in_folder(folder_path):
        image_count = 0
        image_names = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                image_count += 1
                image_names.append(int(file_name.split('.')[0]))
        image_names.sort()
        return image_count, image_names

    for folder in folder_paths:
        folder_path = f'{dataset_path}/frames/' + folder
        csv_path = f'{dataset_path}/csv/' + folder + '.csv'
        all_dirs = []
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                all_dirs.append(os.path.join(root, dir))

        label = list()
        save_path = list()
        frame_counts = list()
        frame_seq_counts = list()
        content_paths = list()
        str_labels = list()

        for video_path in all_dirs:
            frame_paths = glob(video_path + '/*')
            temp_frame_count, temp_frame_seqs = count_images_in_folder(video_path)
            if temp_frame_count == 0:
                continue

            for frame in frame_paths:
                content_path = frame.split('/')[1:-1]
                content_path = '/'.join(content_path)
                content_path = f'{dataset_path}/' + content_path
                frame_path = frame.split('/')[1:]
                frame_path = '/'.join(frame_path)
                frame_path = f'{dataset_path}/' + frame_path

                print(content_path, frame_path)
                if is_real == True:
                    label.append(str(0))
                    str_labels.append('Real Video')
                elif is_real == False :
                    label.append(str(1))
                    str_labels.append('AI Video')
                frame_counts.append(int(temp_frame_count))
                frame_seq_counts.append(temp_frame_seqs)
                save_path.append(frame_path)
                content_paths.append(content_path)
                break

        dic={
            'content_path': Series(data=content_paths),
            'image_path': Series(data=save_path),
            'type_id': Series(data=str_labels),
            'label': Series(data=label),
            'frame_len': Series(data=frame_counts),
            'frame_seq': Series(data=frame_seq_counts)
        }

        print(dic)
        pd.DataFrame(dic).to_csv(csv_path, encoding='utf-8', index=False)

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Got: {v}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process dataset with mandatory --is-real.')
    parser.add_argument('--is-real', type=str2bool, required=True,
                        help="Specify whether video is real: --is-real True or --is-real False (required)")
    parser.add_argument('--dataset-path', type=str, default='GenVideo',
                        help="Path to the dataset directory ")
    parser.add_argument('--folders', nargs='+',
                        help="List of testsets to process ")
    args = parser.parse_args()

    main(args.is_real, args.dataset_path, args.folders)
