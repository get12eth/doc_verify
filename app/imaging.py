"""Decode uploaded images / PDFs into BGR uint8 arrays, safely."""
from __future__ import annotations

import io
import math
import cv2
import pymupdf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidDocument(ValueError):
    """Raised when an upload cannot be decoded or violates a safety limit."""


def load_document(data: bytes, *, page: int = 0, dpi: int = 300, max_pixels: int = 60_000_000) -> np.ndarray:
    if data[:5] == b"%PDF-":
        return _render_pdf(data, page, dpi, max_pixels)
    return _decode_image(data, max_pixels)


def _render_pdf(data: bytes, page: int, dpi: int, max_pixels: int) -> np.ndarray:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types
        raise InvalidDocument("Unreadable PDF.") from exc
    with doc:
        if doc.needs_pass:
            raise InvalidDocument("Password-protected PDFs are not supported.")
        if page == -1:
            pages = list(range(doc.page_count))
        else:
            if not 0 <= page < doc.page_count:
                raise InvalidDocument(f"Page {page} out of range (document has {doc.page_count} page(s)).")
            pages = [page]
        rendered: list[np.ndarray] = []
        for pno in pages:
            pg = doc[pno]
            scale = dpi / 72
            area = pg.rect.width * pg.rect.height * scale * scale
            if area > max_pixels:
                scale *= math.sqrt(max_pixels / area)
            pix = pg.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            rendered.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if len(rendered) == 1:
            return rendered[0]
        # Stitch all pages vertically so a multi-page/merged document is reviewed as one image.
        widths = [img.shape[1] for img in rendered]
        target = max(widths)
        resized = []
        for img in rendered:
            if img.shape[1] != target:
                scale_w = target / img.shape[1]
                img = cv2.resize(img, (target, int(img.shape[0] * scale_w)))
            resized.append(img)
        return np.vstack(resized)


def _decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > max_pixels:
                raise InvalidDocument("Image dimensions exceed the allowed limit.")
            im = ImageOps.exif_transpose(im)  # honour phone-camera rotation
            if im.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white (signature PNGs)
                im = im.convert("RGBA")
                background = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(background, im)
            rgb = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidDocument("Unsupported or corrupt image.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
