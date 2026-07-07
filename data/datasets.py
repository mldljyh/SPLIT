
import os
import re
import pandas as pd
import albumentations
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]
    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]
    return cropped_image


def get_number_from_filename(filename):
    match = re.match(r'(\d+)', filename)  
    if match:
        return int(match.group(1))  
    return float('inf')  


def read_video(folder_path, trans):
    frames = []
    image_paths = sorted(os.listdir(folder_path), key=get_number_from_filename)
    total_frames = len(image_paths)

    if total_frames < 8:
        print(f"No enough frames found in {folder_path}.")
        raise ValueError(f"No enough frames found in {folder_path}.")

    set_frame = 8 if total_frames < 16 else 16
    max_frame = min(set_frame, total_frames)
    for i in range(max_frame):
        image_path = os.path.join(folder_path, image_paths[i])
        image = cv2.imread(image_path)
        image = crop_center_by_percentage(image, 0.1)
        augmented = trans(image=image)
        image = augmented["image"]
        frames.append(image.transpose(2, 0, 1)[np.newaxis, :])

    frames = np.concatenate(frames, 0)
    frames = torch.tensor(frames[np.newaxis, :]).squeeze(0)
    return frames


def set_preprocessing(aug_type, aug_quality):
    aug_list = []
    aug_list.append(albumentations.Resize(224, 224))
    if aug_type == 'Gaussian_blur':
        aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 7),sigma_limit=(aug_quality, aug_quality),p=1.0)) 
    if aug_type == 'JEPG_compression':
        aug_list.append(albumentations.ImageCompression(quality_lower=aug_quality, quality_upper=aug_quality))
    aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
    return albumentations.Compose(aug_list)


class SPLIT_dataset_AP(Dataset):

    def __init__(
        self,
        real_csv,
        fake_csv,
        max_len=9999999,
        aug_type=None,
        aug_quality=None,
        sample_real_to_fake=False,
        seed=42,
    ):
        super(SPLIT_dataset_AP, self).__init__()
        df_real = pd.read_csv(real_csv).head(max_len)
        df_fake = pd.read_csv(fake_csv).head(max_len)
        df_real = self._filter_insufficient_frames(df_real)
        df_fake = self._filter_insufficient_frames(df_fake)
        if sample_real_to_fake:
            if len(df_real) < len(df_fake):
                print("Real set must be larger than fake set for sampling.")
                raise ValueError("Real set must be larger than fake set for sampling.")
            df_real = df_real.sample(n=len(df_fake), random_state=seed).reset_index(drop=True)
            df_fake = df_fake.reset_index(drop=True)
        self.df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        self.trans = set_preprocessing(aug_type, aug_quality)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.loc[index]['label']
        frame_path = self.df.loc[index]['content_path']
        frames = read_video(frame_path,trans=self.trans)
        return frames, label

    @staticmethod
    def _filter_insufficient_frames(df, min_frames=8):
        content_paths = df['content_path'].tolist()
        keep_mask = []
        for path in content_paths:
            try:
                keep_mask.append(len(os.listdir(path)) >= min_frames)
            except FileNotFoundError:
                keep_mask.append(False)
        return df[keep_mask].reset_index(drop=True)
