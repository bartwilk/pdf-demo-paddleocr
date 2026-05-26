"""Extract text from PDFs in sample/, writing one .md per PDF to output/.

Digital pages (native text layer / fillable form fields) are read directly
with PyMuPDF. Scanned-image pages are rasterized and parsed with the
PaddleOCR-VL-1.5 vision-language pipeline on the GPU, which emits structured
markdown (reading order, tables, formulas) directly. OCR'd pages also get an
annotated debug image in image_dir/vis/.

See README.md for the pipeline overview.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Configuration ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Use a project-local model cache instead of ~/.paddlex: PaddleOCR downloads
# the weights here on first run, then later runs reuse them offline. Must be
# set before paddle/paddleocr is imported. Override PADDLE_PDX_CACHE_HOME to
# use a different cache.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / "models"))

import fitz  # PyMuPDF  # noqa: E402
import numpy as np  # noqa: E402

SAMPLE_DIR = ROOT / "sample"
OUTPUT_DIR = ROOT / "output"
VIS_DIR = ROOT / "image_dir" / "vis"

NATIVE_TEXT_MIN_CHARS = 40   # min non-whitespace chars to trust the text layer
OCR_DPI = 300                # rasterization DPI for OCR'd pages
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
    """Build the PaddleOCR-VL-1.5 document-parsing pipeline once, pinned to GPU."""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCRVL

        # VL is multilingual (auto-detected) and reconstructs reading order,
        # tables and formulas itself, so the classic lang/orientation knobs
        # and our manual reading_order() are no longer needed. pipeline_version
        # "v1.5" is the current default; pin it explicitly for reproducibility.
        _OCR = PaddleOCRVL(device="gpu:0", pipeline_version="v1.5")
    return _OCR


def ocr_page(img_rgb: np.ndarray, vis_path: Path | None) -> str:
    """Parse a rasterized page with PaddleOCR-VL; return structured markdown."""
    ocr = get_ocr()
    # PaddleX pipelines consume BGR ndarrays (cv2 convention).
    results = list(ocr.predict(img_rgb[:, :, ::-1]))
    if not results:
        return ""

    res = results[0]

    if SAVE_VIS and vis_path is not None:
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        # res.img holds the pipeline's visualizations keyed by stage
        # (layout_det_res, overall_ocr_res, ...). Save the OCR overlay if
        # present, else whichever image the pipeline produced, under our own
        # basename rather than the auto-generated timestamp name.
        imgs = res.img or {}
        pil = imgs.get("overall_ocr_res") or next(iter(imgs.values()), None)
        if pil is not None:
            pil.convert("RGB").save(vis_path)

    # res.markdown carries the reading-ordered markdown for the page.
    md = res.markdown
    if isinstance(md, dict):
        return (md.get("markdown_texts") or "").rstrip()
    return (getattr(md, "markdown_texts", None) or str(md)).rstrip()


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

        header = f"## Page {i} of {n_pages} [{mode}]"
        chunks.append(f"{header}\n\n{body}".rstrip())
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
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
