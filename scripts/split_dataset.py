import os
import random
import shutil
from pathlib import Path

def split_dataset(src_root="dataset", dst_root="data_split", seed=42, ratios=(3,1,1)):
    random.seed(seed)
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    classes = [p for p in src_root.iterdir() if p.is_dir()]
    if not classes:
        raise FileNotFoundError(f"No class folders found under: {src_root.resolve()}")

    # create dirs
    for split in ["train", "val", "test"]:
        for cls in classes:
            (dst_root / split / cls.name).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        files = [p for p in cls.iterdir() if p.is_file()]
        if len(files) < sum(ratios):
            raise ValueError(f"Class '{cls.name}' has {len(files)} files, but need at least {sum(ratios)}.")
        random.shuffle(files)

        train_files = files[:ratios[0]]
        val_files   = files[ratios[0]:ratios[0]+ratios[1]]
        test_files  = files[ratios[0]+ratios[1]:ratios[0]+ratios[1]+ratios[2]]

        for p in train_files:
            shutil.copy2(p, dst_root / "train" / cls.name / p.name)
        for p in val_files:
            shutil.copy2(p, dst_root / "val" / cls.name / p.name)
        for p in test_files:
            shutil.copy2(p, dst_root / "test" / cls.name / p.name)

    print(f"[OK] Split done. Output -> {dst_root.resolve()}")

if __name__ == "__main__":
    split_dataset()
