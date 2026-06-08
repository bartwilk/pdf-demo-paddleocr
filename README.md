# pdf-demo-paddleocr

Extract text from PDF files. Handles both digital PDFs (native text layer
or fillable form fields) and scanned-image PDFs (GPU document parsing via the
PaddleOCR-VL-1.5 vision-language model).

For each PDF in `sample/`, one `.md` is written to `output/` with the
same basename. OCR'd pages also produce a debug JPG (the pipeline's
annotated visualization) in `image_dir/vis/`.

## Layout
```
sample/      input PDFs
output/      extracted .md files (one per PDF)
image_dir/   PaddleOCR debug visualizations (vis/ subfolder, one .jpg per OCR'd page)
models/      PaddleOCR-VL model + font cache (downloaded on first run, see below)
main.py      extraction script
```

## How it works
For each page in each PDF under `sample/`:
- Read the native text layer with PyMuPDF and measure how much of the page
  area is covered by embedded raster images.
- Keep the text-layer output -- fast and exact -- only when the page is
  essentially all text: at least `NATIVE_TEXT_MIN_CHARS` (40) non-whitespace
  characters **and** less than `IMG_COVER_MIN` (5%) image coverage.
- Otherwise rasterize at `OCR_DPI` (300) and parse the page with the
  PaddleOCR-VL-1.5 pipeline on the GPU. VL is a 0.9B vision-language model
  that emits structured markdown directly -- reading order, tables and
  formulas are reconstructed by the model, so no manual word re-ordering is
  needed.

This means a page is OCR'd when it has no usable text layer **or** when it
hides meaningful content inside images -- a scanned form placed as one big
image behind a thin footer text layer, or a photo addendum with only field
labels in the text layer. OCR'ing the whole rendered page captures both the
printed text and the image content together. Pages that are genuinely all
text (e.g. a filled digital form with only a small logo) stay on the fast
text-layer path.

Each page in the output is prefixed with a markdown heading
`## Page N of M [text-layer|ocr]`.

For OCR'd pages, the pipeline's annotated visualization is written to
`image_dir/vis/<basename>_pNNN.jpg` (controlled by `SAVE_VIS=True` in
`main.py`). To skip writing visualizations, set `SAVE_VIS=False`.

## Requirements
- Linux (or WSL on Windows).
- NVIDIA GPU with CUDA available to PaddlePaddle (the script asserts
  `paddle.device.is_compiled_with_cuda()` before initializing the pipeline).
  PaddleOCR-VL-1.5 needs compute capability >= 7.0 and CUDA 11.8+.
- Python 3.11, PyMuPDF 1.27, `paddleocr[doc-parser]` 3.2.1+,
  paddlepaddle-gpu 3.2.1+. Pinned versions are in `requirements.txt`.
- The `libgomp1` system package (GNU OpenMP runtime). PaddlePaddle's native
  core links against `libgomp.so.1`; without it `import paddle` fails with
  `ImportError: libgomp.so.1: cannot open shared object file`. Install it with
  your distro's package manager (it is not a pip dependency):

  ```bash
  sudo apt-get update && sudo apt-get install -y libgomp1   # Debian/Ubuntu/WSL
  # check it is visible to the loader:
  ldconfig -p | grep libgomp
  ```
- `ccache` (optional). PaddlePaddle JIT-compiles some custom CUDA/C++ ops at
  runtime and uses `ccache` to cache recompiles. Without it you get a harmless
  `UserWarning: No ccache found ...` and slower recompiles -- nothing breaks.
  Install it to silence the warning:

  ```bash
  sudo apt-get install -y ccache   # Debian/Ubuntu/WSL
  which ccache && ccache --version
  ```

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
# system library PaddlePaddle's native core needs, plus optional ccache
# to silence the "No ccache found" recompile warning (see Requirements)
sudo apt-get update && sudo apt-get install -y libgomp1 ccache

pip install -r requirements.txt
pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

Then run:

```bash
python main.py
```

Drop input PDFs into `sample/` before running. Extracted markdown lands in
`output/`, debug visualizations in `image_dir/vis/`.

To run the OCR pipeline on the CPU instead of the GPU (slower, but no CUDA
GPU required — useful on machines without a supported NVIDIA card), pass
`--cpu`:

```bash
python main.py --cpu
```

With `--cpu` the GPU assertion is skipped and the PaddleOCR-VL-1.5 pipeline is
pinned to the CPU. Note this still needs a working PaddlePaddle install; the
CPU-only build comes from plain PyPI (`pip install paddlepaddle==3.3.1`).

### Model cache (downloaded on first run)
The PaddleOCR-VL model weights are **not** committed to the repo. On the
first run PaddleOCR downloads them into the project-local cache at `models/`,
so that first run needs internet access; every later run reuses the cache and
runs with no network. `main.py` points PaddleX at this folder by setting
`PADDLE_PDX_CACHE_HOME` to `models/` before importing PaddleOCR.

After the first run, `models/official_models/` holds the models this pipeline
loads:

| Model | Role |
| --- | --- |
| `PaddleOCR-VL-1.5` | 0.9B vision-language recognizer (text, tables, formulas) |
| `PP-DocLayoutV2` | layout detection / region ordering |

`models/fonts/` holds the fonts PaddleOCR uses to render the debug
visualizations. Everything under `models/` — the downloaded weights plus the
volatile `func_ret/`, `locks/`, `temp/` runtime caches — is git-ignored.

To use a different cache location instead (e.g. the default `~/.paddlex`),
set `PADDLE_PDX_CACHE_HOME` in the environment before running; the in-script
default only applies when the variable is unset. VL is multilingual, so there
is no per-language recognizer to swap -- it auto-detects the script.

### WSL note
On WSL the CUDA driver library lives in a nonstandard path. If you see
`libcuda.so: cannot open shared object file`, point the loader at it:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib python main.py
```

Paddle still finds the driver without this (the warning is non-fatal), but
setting it silences the noise.

## Notes
PaddleOCR-VL-1.5 is robust to skew, warping, screen photography and
low-contrast print, and recovers tables/formulas as markdown. Output is
markdown rather than flat text, so tables render as markdown/HTML and
headings are preserved.

Standalone GPU inference is convenient but not the fastest path. For higher
throughput PaddleOCR-VL can delegate the VL step to a dedicated inference
server (vLLM / SGLang / FastDeploy) -- pass `vl_rec_backend="vllm-server"`
and `vl_rec_server_url=...` to `PaddleOCRVL()` in `get_ocr()`. The layout
detection still runs locally.
