import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm

from data.datasets import SPLIT_dataset_AP, read_video, set_preprocessing
from models import SPLIT_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSVVideoDataset(Dataset):
    def __init__(self, df, aug_type=None, aug_quality=None):
        super(CSVVideoDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.trans = set_preprocessing(aug_type, aug_quality)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        frame_path = self.df.loc[index]["content_path"]
        frames = read_video(frame_path, trans=self.trans)
        return frames, frame_path


def load_and_filter_csvs(csv_paths, max_len=9999999):
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path).head(max_len)
        dfs.append(df)
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    return SPLIT_dataset_AP._filter_insufficient_frames(merged)


def compute_scores(model, df):
    dataset = CSVVideoDataset(df)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    scores = []
    with torch.no_grad():
        for batch_frames, _paths in tqdm(loader, desc="Scoring", leave=False):
            batch_inputs = batch_frames.cuda()
            batch_scores = model.forward_score(batch_inputs)
            scores.extend(batch_scores.cpu().flatten().numpy())
    return np.array(scores)


def evaluate_split(real_pred, fake_pred):
    y_true = np.concatenate([np.zeros(len(real_pred)), np.ones(len(fake_pred))])
    y_pred = np.concatenate([real_pred, fake_pred])

    ap_real = average_precision_score(1 - y_true, -y_pred)
    ap_fake = average_precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    mAP = (ap_real + ap_fake) / 2.0

    precision, recall, _thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    best_precision = float(precision[best_idx])
    best_recall = float(recall[best_idx])

    real_scores = y_pred[y_true == 0]
    fake_scores = y_pred[y_true == 1]

    def fake_recall_at_fpr(target_fpr):
        threshold = np.quantile(real_scores, 1 - target_fpr, method="higher")
        return float((fake_scores >= threshold).mean())

    fake_recall_at_fpr_map = {
        0.001: fake_recall_at_fpr(0.001),
        0.01: fake_recall_at_fpr(0.01),
        0.05: fake_recall_at_fpr(0.05),
    }

    return {
        "ap_real": ap_real,
        "ap_fake": ap_fake,
        "mAP": mAP,
        "roc_auc": roc_auc,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "fake_recall_at_fpr": fake_recall_at_fpr_map,
        "total_samples": len(y_true),
    }


def format_result(args, fake_csvs, metrics):
    return (
        f"Evaluation Results\n"
        f"Encoder: {args.encoder}\n"
        f"Real CSVs: {', '.join(args.real_csv)}\n"
        f"Fake CSVs: {', '.join(fake_csvs)}\n"
        f"Total Samples: {metrics['total_samples']}\n"
        f"AP Real: {metrics['ap_real']:.4f}\n"
        f"AP Fake: {metrics['ap_fake']:.4f}\n"
        f"mAP: {metrics['mAP']:.4f}\n"
        f"ROC AUC: {metrics['roc_auc']:.4f}\n"
        f"Precision (Fake, best F1): {metrics['best_precision']:.4f}\n"
        f"Recall (Fake, best F1): {metrics['best_recall']:.4f}\n"
        f"Fake Recall @ FPR 0.1%: {metrics['fake_recall_at_fpr'][0.001]:.4f}\n"
        f"Fake Recall @ FPR 1%: {metrics['fake_recall_at_fpr'][0.01]:.4f}\n"
        f"Fake Recall @ FPR 5%: {metrics['fake_recall_at_fpr'][0.05]:.4f}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for SPLIT.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--gpu-id", type=str, default="0",
                        help='CUDA GPU device ID(s), e.g., "0" or "1,2,3" (default: "0")')
    parser.add_argument("--encoder", type=str, default="XCLIP-16",
                        help="Encoder model name (default: XCLIP-16)",
                        choices=["CLIP-16", "CLIP-32", "XCLIP-16", "XCLIP-32", "DINO-base", "DINO-large", "ResNet-18", "VGG-16", "EfficientNet-b4", "MobileNet-v3"])
    parser.add_argument("--real-csv", type=str, nargs="+", required=True,
                        help="Path(s) to real data CSV file(s)")
    parser.add_argument("--fake-csv", type=str, nargs="+", required=True,
                        help="Path(s) to fake/synthetic data CSV file(s)")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f"Starting evaluation for {args.encoder}")
    print(f"Real CSVs: {', '.join(args.real_csv)}")
    print(f"Fake CSVs: {', '.join(args.fake_csv)}")

    model = SPLIT_model(encoder_type=args.encoder, loss_type="l2").cuda()
    model.eval()

    real_df = load_and_filter_csvs(args.real_csv)
    print(f"Real samples: {len(real_df)}")
    real_pred = compute_scores(model, real_df)

    fake_df = load_and_filter_csvs(args.fake_csv)
    print(f"Fake samples: {len(fake_df)}")
    fake_pred = compute_scores(model, fake_df)

    metrics = evaluate_split(real_pred, fake_pred)
    result_block = format_result(args, args.fake_csv, metrics)

    print("\n" + "=" * 50)
    print(result_block.strip())
    print("=" * 50)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/result_one_{timestamp}.txt"

    with open(output_file, "w") as f:
        f.write(result_block)
        f.write("\n")

    print(f"\nResults saved to {output_file}")
