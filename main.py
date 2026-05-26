"""Extract text from PDFs in sample/, writing one .txt per PDF to output/.

Digital pages (native text layer / fillable form fields) are read directly
with PyMuPDF. Scanned-image pages are rasterized and OCR'd with PaddleOCR
(PP-OCRv5) on the GPU. OCR'd pages also get an annotated debug image in
image_dir/vis/.

This is the PaddleOCR port of the MMOCR pipeline described in README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

# --- Configuration ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample"
OUTPUT_DIR = ROOT / "output"
VIS_DIR = ROOT / "image_dir" / "vis"

NATIVE_TEXT_MIN_CHARS = 40   # min non-whitespace chars to trust the text layer
OCR_DPI = 300                # rasterization DPI for OCR'd pages
OCR_LANG = "en"              # PaddleOCR recognition language
SAVE_VIS = True              # write annotated JPGs for OCR'd pages

# Set lazily by get_ocr() so the model loads only if an OCR page is hit.
_OCR = None


def assert_gpu() -> None:
    """Fail fast (like the README's torch.cuda assert) if no CUDA GPU."""
    import paddle

    if not paddle.device.is_compiled_with_cuda():
        sys.exit(
            "This paddlepaddle build has no CUDA support. Install the GPU "
            "build matching your CUDA toolkit, e.g.:\n"
            "  pip install paddlepaddle-gpu \\\n"
            "    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
        )
    if paddle.device.cuda.device_count() < 1:
        sys.exit("No CUDA device visible to paddle (check nvidia-smi / drivers).")

    name = paddle.device.cuda.get_device_properties(0).name
    print(f"[gpu] using CUDA device 0: {name}", flush=True)


def get_ocr():
    """Build the PaddleOCR pipeline once, pinned to GPU."""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR

        _OCR = PaddleOCR(
            lang=OCR_LANG,
            device="gpu:0",
            use_doc_orientation_classify=False,  # pages are already upright
            use_doc_unwarping=False,
            use_textline_orientation=True,       # handle rotated text lines
        )
    return _OCR


def reading_order(items: list[tuple[str, np.ndarray]]) -> str:
    """Re-order recognized words into top-to-bottom, left-to-right text.

    `items` is a list of (text, polygon) pairs, where polygon is an array of
    corner points [[x, y], ...]. Words are grouped into lines by vertical
    overlap, then sorted left-to-right within each line.
    """
    boxes = []
    for text, poly in items:
        poly = np.asarray(poly, dtype=float)
        xs, ys = poly[:, 0], poly[:, 1]
        boxes.append(
            {
                "text": text,
                "x": float(xs.min()),
                "cy": float((ys.min() + ys.max()) / 2),
                "h": float(ys.max() - ys.min()),
            }
        )
    if not boxes:
        return ""

    median_h = float(np.median([b["h"] for b in boxes]))
    thresh = max(median_h * 0.6, 1.0)

    boxes.sort(key=lambda b: b["cy"])
    lines: list[list[dict]] = [[boxes[0]]]
    ref = boxes[0]["cy"]
    for b in boxes[1:]:
        if abs(b["cy"] - ref) <= thresh:
            lines[-1].append(b)
        else:
            lines.append([b])
        ref = sum(x["cy"] for x in lines[-1]) / len(lines[-1])

    out_lines = []
    for line in lines:
        line.sort(key=lambda b: b["x"])
        out_lines.append(" ".join(b["text"] for b in line))
    return "\n".join(out_lines)


def ocr_page(img_rgb: np.ndarray, vis_path: Path | None) -> str:
    """Run PaddleOCR on a rasterized page; return text in reading order."""
    ocr = get_ocr()
    # PaddleOCR consumes BGR ndarrays (cv2 convention).
    results = ocr.predict(img_rgb[:, :, ::-1])
    if not results:
        return ""

    res = results[0]
    texts = res.get("rec_texts", [])
    polys = res.get("rec_polys")
    if polys is None:
        polys = res.get("dt_polys", [])

    if SAVE_VIS and vis_path is not None:
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        # res.img is {"ocr_res_img": PIL.Image}; save it under our own
        # basename rather than PaddleOCR's auto-generated timestamp name.
        res.img["ocr_res_img"].convert("RGB").save(vis_path)

    return reading_order(list(zip(texts, polys)))


def extract_pdf(pdf_path: Path) -> str:
    """Extract text from every page of one PDF, with per-page headers."""
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    chunks = []

    for i, page in enumerate(doc, start=1):
        native = page.get_text("text")
        if len(native.split()) and len("".join(native.split())) >= NATIVE_TEXT_MIN_CHARS:
            mode = "text-layer"
            body = native.rstrip()
        else:
            mode = "ocr"
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA -> RGB
                img = img[:, :, :3]
            elif pix.n == 1:  # grayscale -> RGB
                img = np.repeat(img, 3, axis=2)
            vis_path = VIS_DIR / f"{pdf_path.stem}_p{i:03d}.jpg"
            body = ocr_page(img, vis_path)

        header = f"--- Page {i} of {n_pages}  [{mode}] ---"
        chunks.append(f"{header}\n{body}".rstrip())
        print(f"  page {i}/{n_pages}: {mode}", flush=True)

    doc.close()
    return "\n\n".join(chunks) + "\n"


def main() -> None:
    assert_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(SAMPLE_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {SAMPLE_DIR}")

    for pdf_path in pdfs:
        print(f"[pdf] {pdf_path.name}", flush=True)
        text = extract_pdf(pdf_path)
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
