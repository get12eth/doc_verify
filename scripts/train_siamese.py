"""Train a writer-independent Siamese signature embedder and export it to ONNX.

Dataset layout (any number of writers, >= 2 genuine signatures each):

    DATA/<writer_id>/genuine/*.png|jpg
    DATA/<writer_id>/forged/*.png|jpg        optional; skilled forgeries are used as hard negatives

Writers are split into train / validation groups, so the reported EER is on people the model has
never seen, which is the situation in production. Preprocessing is `app.signature.prepare`, exactly
what the API runs at serving time, so there is no train/serve skew.

    python -m scripts.train_siamese --data DATA --out models/signature_siamese.onnx --epochs 30

Then:  DOCVERIFY_SIG_SIAMESE_MODEL=models/signature_siamese.onnx  and re-run scripts/calibrate_signature.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from app.signature import NET_H, NET_W, prepare

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
Writers = dict[str, dict[str, list[np.ndarray]]]


# --------------------------------------------------------------------------- data
def load_writers(root: Path) -> Writers:
    writers: Writers = {}
    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if prepared is not None:
                entry[kind].append(prepared.net_input)
        else:
            skipped += 1
        entry: dict[str, list[np.ndarray]] = {"genuine": [], "forged": []}
        for kind in entry:
            for f in sorted((wdir / kind).glob("*")):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                img = cv2.imread(str(f))
                prepared = prepare(img) if img is not None else None
                if prepared is not None:
                    entry[kind].append(prepared.net_input)
        if len(entry["genuine"]) >= 2:
            writers[wdir.name] = entry
    print(f"loaded {len(writers)} writers; skipped {skipped} unreadable or blank images")
    return writers


def augment(x: np.ndarray) -> np.ndarray:
    """Small random affine jitter (rotation, scale, shear, shift) mimicking natural signing variation."""
    h, w = x.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-6, 6), random.uniform(0.9, 1.1))
    m[0, 1] += random.uniform(-0.15, 0.15)
    m[:, 2] += (random.uniform(-0.04, 0.04) * w, random.uniform(-0.06, 0.06) * h)
    return cv2.warpAffine(x, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]


class TrainPairs(Dataset):
    """Balanced random pairs: 50% same writer (label 1); 50% forgery or other writer (label 0)."""

    def __init__(self, writers: Writers, length: int) -> None:
        self.writers, self.ids, self.length = writers, sorted(writers), length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int):
        wid = random.choice(self.ids)
        w = self.writers[wid]
        a, p = random.sample(w["genuine"], 2)
        if random.random() < 0.5:
            b, label = p, 1.0
        else:
            label = 0.0
            if w["forged"] and (len(self.ids) < 2 or random.random() < 0.5):
                b = random.choice(w["forged"])                       # hard negative: skilled forgery
            else:
                other = random.choice([i for i in self.ids if i != wid])
                b = random.choice(self.writers[other]["genuine"])    # easy negative: different writer
        return to_tensor(augment(a)), to_tensor(augment(b)), torch.tensor(label)


class ValPairs(Dataset):
    """Fixed, un-augmented validation pairs (seeded so epochs are comparable)."""

    def __init__(self, writers: Writers, per_writer: int = 40, seed: int = 0) -> None:
        rng, ids = random.Random(seed), sorted(writers)
        self.pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
        for wid in ids:
            w = writers[wid]
            for _ in range(per_writer):
                a, p = rng.sample(w["genuine"], 2)
                self.pairs.append((a, p, 1.0))
                if w["forged"] and (len(ids) < 2 or rng.random() < 0.5):
                    neg = rng.choice(w["forged"])
                elif len(ids) > 1:
                    neg = rng.choice(writers[rng.choice([i for i in ids if i != wid])]["genuine"])
                else:
                    continue
                self.pairs.append((a, neg, 0.0))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        a, b, y = self.pairs[i]
        return to_tensor(a), to_tensor(b), torch.tensor(y)


# --------------------------------------------------------------------------- model / loss
def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class SigNet(nn.Module):
    """Small VGG-style CNN -> L2-normalised embedding. Input (B, 1, 96, 192) in [0, 1]."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64), conv_block(64, 128),
                                      conv_block(128, 192), nn.AdaptiveAvgPool2d((3, 6)))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(192 * 3 * 6, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.features(x)), dim=1)


def contrastive_loss(ea: torch.Tensor, eb: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    d = torch.linalg.vector_norm(ea - eb, ord=2, dim=1)           # Euclidean distance between embeddings
    return (y * d.pow(2) + (1 - y) * F.relu(margin - d).pow(2)).mean()


# --------------------------------------------------------------------------- evaluation
def eer_from_scores(sims: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the similarity threshold where it occurs."""
    genuine, impostor = sims[labels == 1], sims[labels == 0]
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("EER requires both genuine and impostor validation pairs; add validation writers or forged samples.")
    thresholds = np.unique(sims)
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thresholds[i])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    sims, labels = [], []
    for a, b, y in loader:
        distance = torch.linalg.vector_norm(model(a.to(device)) - model(b.to(device)), ord=2, dim=1)
        sims += (1.0 - distance / 2.0).cpu().tolist()
        labels += y.tolist()
    return eer_from_scores(np.array(sims), np.array(labels))


# --------------------------------------------------------------------------- export
def export_onnx(model: nn.Module, out: Path) -> None:
    import onnxruntime as ort

    model.eval().cpu()
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(2, 1, NET_H, NET_W)
    kwargs = dict(input_names=["input"], output_names=["embedding"], opset_version=17,
                  dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)   # torch >= 2.5
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **kwargs)
    # Parity check: ONNX Runtime must reproduce the PyTorch embeddings.
    ref = model(dummy).detach().numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"input": dummy.numpy()})[0]
    print(f"exported {out}  (max |torch - onnx| = {np.abs(ref - got).max():.2e})")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("models/signature_siamese.onnx"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-pairs", type=int, default=4000, help="random training pairs per epoch")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of WRITERS held out for validation")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-data", type=Path, default=None, help="folder of signers never used in training")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    writers = load_writers(args.data)
    if len(writers) < 4:
        raise SystemExit(f"Need at least 4 writers with >= 2 genuine signatures; found {len(writers)}.")
    ids = sorted(writers)
    random.Random(args.seed).shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_w = {i: writers[i] for i in ids[:n_val]}
    train_w = {i: writers[i] for i in ids[n_val:]}
    print(f"writers: {len(train_w)} train / {len(val_w)} validation | device: {device}")

    train_loader = DataLoader(TrainPairs(train_w, args.epoch_pairs), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(ValPairs(val_w), batch_size=128, shuffle=False)

    model = SigNet(args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader))

    best_eer, best_state, best_thr = 1.0, None, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            loss = contrastive_loss(model(a), model(b), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        eer, thr = evaluate(model, val_loader, device)
        flag = ""
        if eer < best_eer:
            best_eer, best_thr, flag = eer, thr, "  *best*"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {total / len(train_loader):.4f}  val EER {eer:.3f} @ cos {thr:.3f}{flag}")

    if best_state is None:
        raise SystemExit("No valid validation checkpoint was produced; check that validation contains both pair classes.")
    model.load_state_dict(best_state)
    
    if args.test_data:
        test_w = load_writers(args.test_data)
        eer, thr = evaluate(model, DataLoader(ValPairs(test_w), batch_size=128), device)
        print(f"hold-out TEST EER on {len(test_w)} unseen writers: {eer:.3f} (cosine threshold ≈ {thr:.3f})")
    print(f"\nbest validation EER on unseen writers: {best_eer:.3f} (cosine threshold ≈ {best_thr:.3f})")
    export_onnx(model, args.out)
    print("Next: set DOCVERIFY_SIG_SIAMESE_MODEL and run scripts/calibrate_signature on your own pairs.")


if __name__ == "__main__":
    main()
