# pdf-demo-paddleocr

Extract text from PDF files. Handles both digital PDFs (native text layer
or fillable form fields) and scanned-image PDFs (GPU OCR via PaddleOCR).

For each PDF in `sample/`, one `.txt` is written to `output/` with the
same basename. OCR'd pages also produce a debug JPG (detection boxes
overlaid on the page) in `image_dir/vis/`.

## Layout
```
sample/      input PDFs
output/      extracted .txt files (one per PDF)
image_dir/   PaddleOCR debug visualizations (vis/ subfolder, one .jpg per OCR'd page)
models/      PaddleOCR model + font cache (downloaded on first run, see below)
main.py      extraction script
```

## How it works
For each page in each PDF under `sample/`:
- Read the native text layer with PyMuPDF.
- If the page has at least `NATIVE_TEXT_MIN_CHARS` (40) non-whitespace
  characters, keep the text-layer output -- fast and exact.
- Otherwise rasterize at `OCR_DPI` (300) and run PaddleOCR (PP-OCRv5) on
  the GPU. Recognized words are re-ordered into top-to-bottom,
  left-to-right reading order before being written.

Each page in the output is prefixed with
`--- Page N of M  [text-layer|ocr] ---`.

For OCR'd pages, PaddleOCR's annotated visualization is written to
`image_dir/vis/<basename>_pNNN.jpg` (controlled by `SAVE_VIS=True` in
`main.py`). To skip writing visualizations, set `SAVE_VIS=False`.

## Requirements
- Linux (or WSL on Windows).
- NVIDIA GPU with CUDA available to PaddlePaddle (the script asserts
  `paddle.device.is_compiled_with_cuda()` before initializing PaddleOCR).
- Python 3.11, PyMuPDF 1.27, PaddleOCR 3.x, paddlepaddle-gpu 3.x.
  Pinned versions are in `requirements.txt`.

`paddlepaddle-gpu` is **not** installable from plain PyPI -- the GPU build
must come from Paddle's CUDA index, matched to your CUDA toolkit. A
Blackwell card (e.g. RTX 5080, compute capability 12.0 / sm_120) needs a
CUDA 12.8+ build; this project was validated with the `cu129` build:

```bash
pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

Use the `cu126` index instead for older (pre-Blackwell) GPUs.

## Running
From the project root, install dependencies once:

```bash
pip install -r requirements.txt
pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

Then run:

```bash
python main.py
```

Drop input PDFs into `sample/` before running. Extracted text lands in
`output/`, debug visualizations in `image_dir/vis/`.

### Model cache (downloaded on first run)
The PP-OCRv5 model weights are **not** committed to the repo. On the first
run PaddleOCR downloads them into the project-local cache at `models/`, so
that first run needs internet access; every later run reuses the cache and
runs with no network. `main.py` points PaddleX at this folder by setting
`PADDLE_PDX_CACHE_HOME` to `models/` before importing PaddleOCR.

After the first run, `models/official_models/` holds the three models this
pipeline loads:

| Model | Role |
| --- | --- |
| `PP-OCRv5_server_det` | text detection |
| `en_PP-OCRv5_mobile_rec` | English recognition (`OCR_LANG="en"`) |
| `PP-LCNet_x1_0_textline_ori` | text-line orientation |

`models/fonts/` holds the fonts PaddleOCR uses to render the debug
visualizations. Everything under `models/` — the downloaded weights plus the
volatile `func_ret/`, `locks/`, `temp/` runtime caches — is git-ignored.

To use a different cache location instead (e.g. the default `~/.paddlex`),
set `PADDLE_PDX_CACHE_HOME` in the environment before running; the in-script
default only applies when the variable is unset. Switching `OCR_LANG` to
another language downloads that language's recognizer into
`models/official_models/` on the next run (internet required once).

### WSL note
On WSL the CUDA driver library lives in a nonstandard path. If you see
`libcuda.so: cannot open shared object file`, point the loader at it:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib python main.py
```

Paddle still finds the driver without this (the warning is non-fatal), but
setting it silences the noise.

## Notes
OCR on photographed / scanned pages is readable but noisy on handwriting
and low-contrast print. For cleaner OCR you can enable PaddleOCR's
document pre-processing in `get_ocr()` -- set
`use_doc_orientation_classify=True` and/or `use_doc_unwarping=True` -- or
pre-process the rasterized image (deskew / contrast / denoise) before
handing it to PaddleOCR. Switching `lang="en"` to another language loads
the matching PP-OCRv5 recognizer.
