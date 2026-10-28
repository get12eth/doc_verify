"""Locate a clear row of three colored-ink specimen signatures on a page."""
from __future__ import annotations

from itertools import combinations
import subprocess

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def _specimen_heading_bottom(image: np.ndarray) -> float | None:
    encoded_ok, encoded = cv2.imencode(".png", image)
    if not encoded_ok:
        return None
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", "eng", "tsv"],
            input=encoded.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    heading_lines: dict[tuple[str, str, str, str], list[tuple[str, int, float]]] = {}
    image_height = image.shape[0]
    for line in result.stdout.decode("utf-8", errors="replace").splitlines()[1:]:
        columns = line.split("\t", 11)
        if len(columns) != 12:
            continue
        text = columns[11].strip().lower()
        if "specimen" not in text and "signature" not in text and "signatory" not in text:
            continue
        try:
            top = int(columns[7])
            height = int(columns[9])
            confidence = float(columns[10])
        except ValueError:
            continue
        key = (columns[1], columns[2], columns[3], columns[4])
        heading_lines.setdefault(key, []).append((text, top + height, confidence))

    candidates = []
    for words in heading_lines.values():
        line_text = " ".join(word[0] for word in words)
        has_specimen = "specimen" in line_text
        has_signature = "signature" in line_text or "signatory" in line_text
        if has_specimen and has_signature:
            candidates.append(max(word[1] for word in words) / image_height)
    if not candidates:
        return None
    return min(candidates)


def find_three_signature_row(image: np.ndarray) -> tuple[Box, Box, Box] | None:
    """Find one compact, evenly spaced ink row directly beneath an OCR specimen heading."""
    height, width = image.shape[:2]
    heading_bottom = _specimen_heading_bottom(image)
    if heading_bottom is None:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ink = cv2.inRange(hsv, np.array([85, 25, 25]), np.array([155, 255, 255]))
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    components: list[Box] = []
    page_area = width * height
    for x, y, box_width, box_height, area in stats[1:count]:
        x0, y0 = x / width, y / height
        x1, y1 = (x + box_width) / width, (y + box_height) / height
        normalized_width, normalized_height = box_width / width, box_height / height
        area_fraction = area / page_area
        aspect = normalized_width / max(normalized_height, 1e-9)
        if (
            heading_bottom + 0.015 <= (y0 + y1) / 2 <= heading_bottom + 0.20
            and
            0.0001 <= area_fraction <= 0.001
            and 0.012 <= normalized_width <= 0.07
            and 0.01 <= normalized_height <= 0.045
            and 0.5 <= aspect <= 3
        ):
            components.append((float(x0), float(y0), float(x1), float(y1)))

    if len(components) < 3:
        return None

    rows = []
    for row in combinations(components, 3):
        row = tuple(sorted(row, key=lambda box: (box[0] + box[2]) / 2))
        centers_x = [(box[0] + box[2]) / 2 for box in row]
        centers_y = [(box[1] + box[3]) / 2 for box in row]
        gaps = [centers_x[1] - centers_x[0], centers_x[2] - centers_x[1]]
        widths = [box[2] - box[0] for box in row]
        heights = [box[3] - box[1] for box in row]
        if min(gaps) < 0.08 or max(gaps) > 0.35 or max(gaps) / min(gaps) > 1.7:
            continue
        if max(centers_y) - min(centers_y) > 0.025:
            continue
        if max(widths) / min(widths) > 2 or max(heights) / min(heights) > 2:
            continue

        spacing_error = abs(gaps[0] - gaps[1]) / max(gaps)
        alignment_error = (max(centers_y) - min(centers_y)) / 0.025
        size_error = (max(widths) / min(widths) - 1) + (max(heights) / min(heights) - 1)
        rows.append((spacing_error + alignment_error + size_error, row))

    if not rows:
        return None
    rows.sort(key=lambda result: result[0])
    best_score, best_row = rows[0]
    if best_score > 2.0:
        return None

    best_centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in best_row]
    for score, row in rows[1:]:
        centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in row]
        same_row = max(abs(a[1] - b[1]) for a, b in zip(best_centers, centers)) < 0.015
        same_trio = max(abs(a[0] - b[0]) for a, b in zip(best_centers, centers)) < 0.03
        if not (same_row and same_trio) and score - best_score < 0.15:
            return None

    return best_row