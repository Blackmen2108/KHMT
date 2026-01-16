import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from preprocessing.utils_3d import resample_image, clip_and_normalize
from experiment_config import config

def preprocess_all_mha(label_csv, spacing=(1.0, 1.0, 1.0)):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = config.PROCESSED_DIR / "image"
    meta_dir = config.PROCESSED_DIR / "metadata"
    img_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    df = pd.read_csv(label_csv)
    done_files = {p.stem for p in img_dir.glob("*.npy")}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["SeriesInstanceUID"]
        mha_path = config.RAW_DIR / f"{uid}.mha"
        out_path = img_dir / f"{uid}.npy"

        if uid in done_files:
            continue  # skip already processed

        img = sitk.ReadImage(str(mha_path))
        img = resample_image(img, new_spacing=spacing)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = clip_and_normalize(arr)

        np.save(out_path, arr)

        meta = {
            "origin": img.GetOrigin(),
            "spacing": img.GetSpacing(),
            "direction": img.GetDirection(),
        }
        np.save(meta_dir / f"{uid}.npy", meta)

if __name__ == "__main__":
    label_csv = config.RAW_DIR / "train.csv"
    preprocess_all_mha(label_csv)
