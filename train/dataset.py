import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CleanUTKFaceDataset(Dataset):
    """
    A cleaned version of the UTKFace dataset that:
    1) Skips images with malformed filenames
    2) Filters out age outliers based on the 1st and 99th percentiles
    3) Discards images with resolution below 64×64
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 1) Collect all image paths
        all_paths = []
        for part in os.listdir(root_dir):
            part_path = os.path.join(root_dir, part)
            if not os.path.isdir(part_path):
                continue
            for file in os.listdir(part_path):
                if not file.lower().endswith(".jpg"):
                    continue
                all_paths.append(os.path.join(part_path, file))

        # 2) Extract age from filenames, skip invalid ones
        valid_paths = []
        bad_format = []
        ages = []
        for p in all_paths:
            fname = os.path.basename(p)
            try:
                age = int(fname.split("_")[0])
                valid_paths.append((p, age))
                ages.append(age)
            except:
                bad_format.append(p)

        if len(bad_format) > 0:
            print(f"[DATASET] Skipped {len(bad_format)} images with malformed filenames.")

        # 3) Filter out age outliers (outside 1st and 99th percentiles)
        if len(ages) > 0:
            ages_np = np.array(ages)
            low_p, high_p = np.percentile(ages_np, [1, 99])
        else:
            low_p, high_p = 0, 100

        filtered = []
        for (p, a) in valid_paths:
            if a < low_p or a > high_p:
                continue
            filtered.append((p, a))
        print(f"[DATASET] {len(filtered)}/{len(valid_paths)} images remained after age outlier filtering.")

        # 4) Remove low-resolution images (min 64×64)
        final = []
        low_res = []
        for (p, a) in filtered:
            try:
                img = Image.open(p)
                if img.width < 64 or img.height < 64:
                    low_res.append(p)
                else:
                    final.append((p, a))
            except:
                low_res.append(p)

        if len(low_res) > 0:
            print(f"[DATASET] Skipped {len(low_res)} images due to low resolution or read failure.")

        # 5) Final cleaned lists of image paths and ages
        self.image_paths = [p for (p, _) in final]
        self.ages = [a for (_, a) in final]

        print(f"[DATASET] Total cleaned samples: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        age = self.ages[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Return normalized age in the range [0, 1]
        return image, torch.tensor(age / 100.0, dtype=torch.float32)
