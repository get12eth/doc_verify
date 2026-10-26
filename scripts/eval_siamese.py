"""Error-rate table for the trained Siamese model on signers it has never seen.

    python -m scripts.eval_siamese data/test_holdout --model models/signature_siamese.onnx

Uses exactly the serving path (app.signature.prepare -> ONNX -> cosine). Prints, per threshold:
  FRR = % of SAME-signer pairs wrongly rejected     FAR = % of DIFFERENT-signer pairs wrongly accepted
Negatives here are other people's genuine signatures ("random forgeries"). Skilled forgeries are NOT
measured, so real-world FAR against a determined imitator will be higher than this table shows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.signature import prepare

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="folder with <signer>/genuine/*.png")
    ap.add_argument("--model", type=Path, default=Path("models/signature_siamese.onnx"))
    ap.add_argument("--negatives", type=int, default=20000, help="random different-signer pairs to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    emb: dict[str, np.ndarray] = {}
    for wdir in sorted(p for p in args.data.iterdir() if p.is_dir()):
        xs = []
        for f in sorted(f for f in (wdir / "genuine").iterdir() if f.suffix.lower() in EXT):
            img = cv2.imread(str(f))
            prepared = prepare(img) if img is not None else None
            if prepared is not None:
                xs.append(prepared.net_input)
        if len(xs) >= 2:
            v = sess.run(None, {in_name: np.stack(xs)[:, None].astype(np.float32)})[0]
            emb[wdir.name] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    if len(emb) < 2:
        raise SystemExit("Need at least 2 signers with >= 2 usable genuine signatures.")

    genuine = np.concatenate([(e @ e.T)[np.triu_indices(len(e), 1)] for e in emb.values()])
    rng, ids = np.random.default_rng(args.seed), list(emb)
    impostor = np.empty(args.negatives)
    for k in range(args.negatives):
        a, b = rng.choice(len(ids), 2, replace=False)
        ea, eb = emb[ids[a]], emb[ids[b]]
        impostor[k] = ea[rng.integers(len(ea))] @ eb[rng.integers(len(eb))]

    print(f"{len(emb)} signers | {genuine.size} same-signer pairs | {impostor.size} different-signer pairs")
    print(f"cosine mean: same {genuine.mean():.3f}, different {impostor.mean():.3f}\n")
    print("threshold   FRR (same rejected)   FAR (different accepted)")
    for t in np.arange(0.50, 0.96, 0.05):
        print(f"  {t:.2f}        {100 * (genuine < t).mean():6.2f}%              {100 * (impostor >= t).mean():6.2f}%")
    ts = np.unique(np.concatenate([genuine, impostor]))
    far = np.array([(impostor >= t).mean() for t in ts])
    frr = np.array([(genuine < t).mean() for t in ts])
    i = int(np.argmin(np.abs(far - frr)))
    print(f"\nEER ≈ {100 * (far[i] + frr[i]) / 2:.2f}% at cosine {ts[i]:.3f}")
    for target in (0.05, 0.01):
        ok = np.nonzero(far <= target)[0]
        if ok.size:
            print(f"FAR <= {target:.0%}: threshold {ts[ok[0]]:.3f}  (FRR there {100 * frr[ok[0]]:.1f}%)")


if __name__ == "__main__":
    main()
