"""Pick signature thresholds from labelled pairs.

Usage:  python -m scripts.calibrate_signature pairs.csv [--target-far 0.01]

pairs.csv columns:  path_a,path_b,label      (label 1 = same writer / genuine, 0 = forgery or different writer)
Use ~100+ pairs of YOUR kind of documents; public sets (CEDAR, BHSig260) are a good starting point.
"""
import argparse
import csv

import cv2
import numpy as np

from app.config import get_settings
from app.embedders import load_embedder
from app.signature import compare_signatures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--target-far", type=float, default=0.01, help="max false-accept rate for the MATCH threshold")
    args = ap.parse_args()

    settings = get_settings()
    embedder = load_embedder(settings)       # calibrates whichever method the API will actually use
    method = embedder.name if embedder else "heuristic"
    scores, labels, skipped = [], [], 0
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            a, b = cv2.imread(row["path_a"]), cv2.imread(row["path_b"])
            if a is None or b is None:
                skipped += 1
                continue
            r = compare_signatures(a, b, settings, embedder)
            if r.score is None:
                skipped += 1
                continue
            scores.append(r.score)
            labels.append(int(row["label"]))

    s, y = np.array(scores), np.array(labels)
    genuine, impostor = s[y == 1], s[y == 0]
    if genuine.size == 0 or impostor.size == 0:
        raise SystemExit("Need both genuine (1) and impostor (0) pairs.")

    thresholds = np.unique(s)
    far = np.array([(impostor >= t).mean() for t in thresholds])   # forgeries wrongly accepted
    frr = np.array([(genuine < t).mean() for t in thresholds])     # genuine wrongly rejected
    eer_i = int(np.argmin(np.abs(far - frr)))
    ok = np.nonzero(far <= args.target_far)[0]
    match_t = float(thresholds[ok[0]]) if ok.size else float(thresholds[-1])

    print(f"method: {method} | pairs used: {s.size} (skipped {skipped})")
    print(f"EER ≈ {(far[eer_i] + frr[eer_i]) / 2:.3f} at threshold {thresholds[eer_i]:.3f}")
    print(f"MATCH threshold for FAR <= {args.target_far:.2%}: {match_t:.3f}  "
          f"(FRR there: {(genuine < match_t).mean():.2%})")
    print(f"Suggested REVIEW threshold (EER point): {thresholds[eer_i]:.3f}")
    prefix = {"heuristic": "DOCVERIFY_SIG", "siamese": "DOCVERIFY_SIG_SIAMESE", "sigverse": "DOCVERIFY_SIG_SIGVERSE"}[method]
    print(f"\nSet {prefix}_MATCH_THRESHOLD and {prefix}_REVIEW_THRESHOLD accordingly.")


if __name__ == "__main__":
    main()
