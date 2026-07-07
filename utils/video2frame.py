import os
from glob import glob
from moviepy.editor import VideoFileClip
import multiprocessing
import math
import random


def get_video_length(file_path):
    video = VideoFileClip(file_path)
    return video.duration

def process_video(video_path, dataset_path):
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[:-1]
    video_name = '.'.join(video_name)

    path = video_path.split('/')[2:-1]
    path = '/'.join(path)
    image_path = f'{dataset_path}/frames/'+path+'/'+ video_name+'/'
    
    if os.path.exists(image_path):
        print(video_name, "frames exist")
    else:
        print(video_name, end='\r')
        try:
            try:
                frame_rate = 8
                duration = 3
                video_length = get_video_length(video_path)
                if video_length <= 3:
                    start_time = 0
                else:
                    start_time = math.floor(random.uniform(0, video_length-3))
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                os.system(f"cd {image_path} | ffmpeg -loglevel quiet -ss {start_time} -t {duration} -i {video_path} -vf fps={frame_rate} {image_path}%d.jpg")
            except Exception as e:
                with open('error.log', 'a') as f:
                    f.write(f"{video_name} error: {str(e)}\n")
                print(f"{video_name} error: {str(e)}\n")
        except Exception as e:
            with open('error.log', 'a') as f:
                f.write(f"{video_name} skipped: {str(e)}\n")
            print(f"{video_name} skipped: {str(e)}\n")

import argparse

if __name__ == '__main__':

    random.seed(42)

    parser = argparse.ArgumentParser(description='Specify the dataset path.')
    parser.add_argument('--dataset-path', type=str, default='datasets', 
                        help='Path to the dataset directory (default: datasets)')
    args = parser.parse_args()
    dataset_path = args.dataset_path

    video_paths = glob(f"{dataset_path}/video/**", recursive=True)
    video_paths = [vp for vp in video_paths if vp.endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif'))]

    print(f"Find {len(video_paths)} videos!")
    args_list = [(vp, dataset_path) for vp in video_paths]

    with multiprocessing.Pool(processes=32) as pool:
        pool.starmap(process_video, args_list)




