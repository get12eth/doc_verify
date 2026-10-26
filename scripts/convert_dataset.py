"""Convert the Hugging Face signature dataset into the folder layout scripts/train_siamese.py expects.

    <out>/<person_id>/genuine/001.png ...

  train + validation  ->  --out       (training pool; the trainer holds out some signers itself)
  test                ->  --test-out  (never used for training; final check via train_siamese --test-data)

What the dataset inspection showed: 315 signers, exactly 24 images each, disjoint signers across
splits, and no forged/genuine marker. 24 per signer equals the genuine count of CEDAR and BHSig260,
so every image is treated as genuine. If a signer ever had more than 24 images, stop and re-check.

    python -m scripts.convert_dataset --out data/train_pool --test-out data/test_holdout
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

REPO = "rakshitdabral/Signature-Verification-Dataset"


def to_rgb(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white so signatures never end up on a black background."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def export_split(split, out_dir: Path, owners: dict[str, str], split_name: str, progress=lambda x, **k: x) -> Counter:
    """Write one split. `owners` maps person -> split, to catch signers appearing in two splits."""
    names = split.features["label"].names
    counts: Counter = Counter()
    for item in progress(split, desc=split_name):
        person = names[item["label"]]
        if owners.setdefault(person, split_name) != split_name:
            raise SystemExit(f"{person} appears in both '{owners[person]}' and '{split_name}': "
                             "splits are not signer-disjoint, so they must not be treated as separate.")
        folder = out_dir / person / "genuine"
        folder.mkdir(parents=True, exist_ok=True)
        counts[person] += 1
        to_rgb(item["image"]).save(folder / f"{counts[person]:03d}.png")
    return counts


def main() -> None:
    from datasets import load_dataset
    from tqdm import tqdm

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/train_pool"))
    ap.add_argument("--test-out", type=Path, default=Path("data/test_holdout"))
    args = ap.parse_args()
    for d in (args.out, args.test_out):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"{d} is not empty; remove it first so images are not duplicated.")

    ds = load_dataset(REPO)
    owners: dict[str, str] = {}
    train_counts = Counter()
    for name in ("train", "validation"):
        train_counts += export_split(ds[name], args.out, owners, name, tqdm)
    test_counts = export_split(ds["test"], args.test_out, owners, "test", tqdm)

    for label, counts, path in (("training pool", train_counts, args.out), ("test hold-out", test_counts, args.test_out)):
        sizes = sorted(set(counts.values()))
        print(f"{label}: {len(counts)} signers, images per signer {sizes} -> {path}")
        if any(n != 24 for n in counts.values()):
            print("  WARNING: not every signer has 24 images; re-check the genuine-only assumption.")


if __name__ == "__main__":
    main()
