import os, re, tarfile, time, traceback, unicodedata, datetime, threading, glob, shutil, uuid, json, atexit, warnings, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import pandas as pd
import pydicom
import pydicom.config as pdc
from difflib import SequenceMatcher
import blosc

# ================= Config =================
pdc.convert_wrong_length_to_UN = True
warnings.filterwarnings("ignore")

INPUT_DIR = os.environ.get(
    "INPUT_DIR",
    "<PRIVATE_DATA_PATH>"
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "<PRIVATE_DATA_PATH>"
)

IMAGES_DIR   = os.path.join(OUTPUT_DIR, "images2")
LOG_DIR      = os.path.join(OUTPUT_DIR, "logs")
TMP_META_DIR = os.path.join(OUTPUT_DIR, "tmp_meta")
TMP_TAR_DIR  = os.path.join(OUTPUT_DIR, "tmp_tar")
ERROR_DIR    = os.path.join(OUTPUT_DIR, "error")

for d in [IMAGES_DIR, LOG_DIR, TMP_META_DIR, TMP_TAR_DIR, ERROR_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
WRITE_POOL  = ThreadPoolExecutor(max_workers=int(os.environ.get("WRITE_THREADS", "4")))
atexit.register(lambda: WRITE_POOL.shutdown(wait=True))
LOG_INTERVAL = 200
blosc.set_nthreads(int(os.environ.get("BLOSC_NTHREADS", str(os.cpu_count() or 8))))
_LZ4HC_CLEVEL = 9

_log_lock = threading.Lock()
_err_lock = threading.Lock()

def log_global(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        print(f"[{ts}] {msg}", flush=True)

def _task_id():
    return os.environ.get("SLURM_ARRAY_TASK_ID", "single")

def _err_csv_path():
    return os.path.join(ERROR_DIR, f"slice_error_{_task_id()}.csv")

def record_bad_slice(tar_name: str, dcm_name: str):
    path = _err_csv_path()
    with _err_lock:
        is_new = not os.path.exists(path)
        with open(path, "a") as f:
            if is_new:
                f.write("tar_name,dcm_name\n")
            f.write(f"{tar_name},{dcm_name}\n")

# ================= Phase =================
PHASE_KEYWORDS = {
    "PRE": ["pre","plain","noncontrast","non-contrast","unenhanced","precontrast","pre_con","pre con"],
    "ART": ["art","arterial","a_phase","art_phase","early"],
    "PV" : ["pv","ven","venous","portal","pvp","ven_phase","portal_venous"],
    "DEL": ["delay","delayed","equilibrium","eq","late","post_180","3min","5min"],
    "POST":["post","postcontrast","contrast","enhanced","post_con","post con"]
}

def _truthy(v):
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s not in ("", "0", "false", "none", "null")

def phase_label(desc: str, has_contrast: bool) -> str:
    if not desc:
        return "UNKNOWN"
    s = desc.lower()
    for ph, kws in PHASE_KEYWORDS.items():
        for kw in kws:
            if kw in s:
                return ph
    if has_contrast:
        return "POST"
    if "pre" in s:
        return "PRE"
    return "UNKNOWN"

# ================= Helpers =================
def safe_get(ds, attr):
    try:
        val = getattr(ds, attr, None)
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)
    except Exception:
        return None

def safe_get_all_tags(ds):
    """
    Best-effort expansion of DICOM metadata into a flat dictionary.
    Complex structures are transformed as JSON strings.
    On failure, returns ({}, False); caller should fill with NaN if needed.
    """
    out = {}
    try:
        for elem in ds.iterall():
            if elem.keyword == "PixelData" or elem.tag == 0x7FE00010:
                continue
            key = elem.keyword if elem.keyword else f"Tag_{int(elem.tag):08X}"
            try:
                val = elem.value
                if isinstance(val, (bytes, bytearray)):
                    val = f"<bytes:{len(val)}>"
                elif isinstance(val, (list, tuple, dict, np.ndarray)):
                    val = json.dumps(val, ensure_ascii=False)
                out[key] = val
            except Exception:
                out[key] = np.nan
    except Exception:
        return {}, False
    return out, True

def _rgb_to_gray_fast(rgb: np.ndarray) -> np.ndarray:
    rgbf = rgb.astype(np.float32, copy=False)
    return (0.2989 * rgbf[..., 0] + 0.5870 * rgbf[..., 1] + 0.1140 * rgbf[..., 2]).astype(np.float32)

def to_int16_volume(vol_f32: np.ndarray) -> np.ndarray:
    """
    Convert pixel value into int16 range before compressing the imaging data
    - If already integer and within int16 range → direct cast
    - If float but already within int16 range → round then cast
    - Otherwise, linearly rescale into [-32768, 32767]
    """
    vol_f32 = np.asarray(vol_f32)
    finite_mask = np.isfinite(vol_f32)
    if not finite_mask.any():
        return np.zeros_like(vol_f32, dtype=np.int16)
    vmin, vmax = np.min(vol_f32[finite_mask]), np.max(vol_f32[finite_mask])
    if np.issubdtype(vol_f32.dtype, np.integer) and vmin >= -32768 and vmax <= 32767:
        return vol_f32.astype(np.int16, copy=False)
    if vmin >= -32768 and vmax <= 32767:
        return np.rint(vol_f32).astype(np.int16)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(vol_f32, dtype=np.int16)
    scaled = (vol_f32 - vmin) / (vmax - vmin) * 65535.0 - 32768.0
    return np.rint(np.clip(scaled, -32768, 32767)).astype(np.int16)


def write_blosc_int16(path: str, vol_i16: np.ndarray):
    '''
    Use os.replace to write the compressed imaging data, make sure there is no imcomplete data
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    packed = blosc.pack_array(
        vol_i16,
        cname="lz4hc",
        clevel=_LZ4HC_CLEVEL,
        shuffle=blosc.BITSHUFFLE,
    )
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(packed)
    os.replace(tmp_path, path)

# ================= Decode =================
def decode_to_gray(ds):
    """
    Convert a DICOM dataset into a 2D grayscale image.

    The function attempts to decode `ds.pixel_array` and reduce it to a usable
    2D slice by:
      - Converting RGB data to grayscale
      - Keeping 2D integer arrays as int16 when possible
      - Casting other 2D arrays to float32
      - For multi-dimensional arrays, recursively selecting a middle slice

    If `.pixel_array` is unavailable, it falls back to decoding raw PixelData.

    Returns:
        A 2D grayscale numpy array, or None if decoding fails.
    """

    def try_make_2d(a):
        if a is None:
            return None
        a = np.squeeze(a)
        if not isinstance(a, np.ndarray) or a.size == 0:
            return None
        # RGB → Gray Scale
        if a.ndim >= 3:
            for axis in range(a.ndim):
                if a.shape[axis] == 3:
                    return _rgb_to_gray_fast(np.moveaxis(a, axis, -1))
        if a.ndim == 2:

            if np.issubdtype(a.dtype, np.integer):
                try:
                    amin = a.min()
                    amax = a.max()
                except Exception:
                    amin, amax = None, None
                if amin is not None and amax is not None and amin >= -32768 and amax <= 32767:
                    return a.astype(np.int16, copy=False)
            return a.astype(np.float32)
    
        if a.ndim >= 3:
            for axis in range(a.ndim):
                if a.shape[axis] > 1:
                    mid = a.shape[axis] // 2
                    sub = np.take(a, mid, axis)
                    out = try_make_2d(sub)
                    if out is not None:
                        return out
        return None

    try:
        arr = getattr(ds, "pixel_array", None)
        g = try_make_2d(arr)
        if g is not None:
            return g
    except Exception:
        pass

    try:
        raw = getattr(ds, "PixelData", None)
        rows, cols = int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))
        if raw and rows > 0 and cols > 0:
            buf = np.frombuffer(raw, dtype=np.uint8)
            if buf.size == rows * cols * 3:
                return _rgb_to_gray_fast(buf.reshape(rows, cols, 3))
            if buf.size == rows * cols:
                return buf.reshape(rows, cols).astype(np.float32)
    except Exception:
        pass
    return None


def safe_write_meta(df: pd.DataFrame, base_prefix: str):
    '''
    Write metadata to a CSV file with a three-level fallback strategy.

    The function attempts:
      1) Standard CSV writing using the C engine.
      2) Fallback to the Python engine with all columns cast to string.
      3) Final fallback: write a header-only CSV with basic columns.

    For each successful write, a corresponding "<file>.ok" marker is created.
    Row/column counts are logged when available.

    This ensures metadata output is always produced, even under unexpected
    encoding or serialization failures.
    '''
    base_dir = os.path.dirname(base_prefix)
    os.makedirs(base_dir, exist_ok=True)

    csv_path = base_prefix + ".csv"
    ok_path  = csv_path + ".ok"


    try:
        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_MINIMAL
        )
        with open(ok_path, "w") as f:
            f.write("OK\n")
        rows, cols = df.shape
        log_global(f"[META] csv write successfully: {csv_path}, rows={rows}, cols={cols}")
        return
    except Exception as e:
        log_global(f"[ERROR] csv(C-engine) write failed for {base_prefix}: {repr(e)}")

 
    try:
        df2 = df.copy()
        for c in df2.columns:
            df2[c] = df2[c].astype("string[python]")
        df2.astype(str).to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
            engine="python"
        )
        with open(ok_path, "w") as f:
            f.write("OK\n")
        rows, cols = df.shape
        log_global(f"[META] csv write successfully (python engine): {csv_path}, rows={rows}, cols={cols}")
        return
    except Exception as e:
        log_global(f"[ERROR] csv(Python-engine) write failed for {base_prefix}: {repr(e)}")


    try:
        basic_cols = ["tar_name", "dcm_name", "phase", "save_name"]
        header_line = ",".join(basic_cols) + "\n"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header_line)
        with open(ok_path, "w") as f:
            f.write("EMPTY\n")
        log_global(f"[ERROR] metadata write degraded to EMPTY header-only csv for {base_prefix}")
    except Exception as e:
        errfile = os.path.join(ERROR_DIR, "meta_write_fatal.txt")
        with open(errfile, "a") as f:
            f.write(f"{datetime.datetime.now()} FATAL meta write error for {base_prefix}: {repr(e)}\n")
        log_global(f"[ERROR] FATAL meta write error for {base_prefix}: {repr(e)}")


def process_one_tar(tar_path: str):
    """
    Process a single DICOM .tar.gz archive and generate slice-level metadata
    and compressed Blosc volumes.

    Behavior:
      - For valid slices: decode to 2D grayscale, group by (H, W), stack into
        H×W×L volumes, and write <stem>_<H>x<W>_<L>.blosc.
      - For unreadable slices: write a 1×1×1 placeholder volume named
        *_broke_1x1x1.blosc with slice_status="bad".
      - If all slices fail or a TAR-level error occurs: write a single
        <stem>_1x1x1_<tar>.blosc placeholder with slice_status="all_bad".

    Returns:
        A list of metadata rows (one per slice), including slice_index,
        phase label, save_name, and slice_status.
    """
    tar_name = os.path.basename(tar_path)
    stem = tar_name[:-7] if tar_name.endswith(".tar.gz") else os.path.splitext(tar_name)[0]
    subdir = os.path.join(IMAGES_DIR, stem)
    os.makedirs(subdir, exist_ok=True)

    metadata_rows = []
    groups = defaultdict(list)
    any_good = False
    bad_slice_count = 0
    total_slice_count = 0

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.lower().endswith(".dcm"):
                    continue
                total_slice_count += 1
                dcm_name = m.name
                try:
                    fo = tar.extractfile(m)
                except Exception:
                    fo = None

                ds = None
                g  = None
                if fo is not None:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ds = pydicom.dcmread(fo, force=True)
                        g = decode_to_gray(ds)
                    except Exception:
                        g = None

                if g is None:
                    bad_slice_count += 1
                    record_bad_slice(tar_name, dcm_name)
                    rel_name = f"{stem}/{os.path.basename(dcm_name)}_broke_1x1x1.blosc"
                    out_path = os.path.join(IMAGES_DIR, rel_name)
                    WRITE_POOL.submit(write_blosc_int16, out_path, np.zeros((1, 1, 1), dtype=np.int16))
                    row = {
                        "tar_name": tar_name,
                        "dcm_name": dcm_name,
                        "slice_index": 0,
                        "phase": "UNKNOWN",
                        "save_name": rel_name,
                        "slice_status": "bad"
                    }
                    metadata_rows.append(row)
                    continue

                any_good = True
                H, W = int(g.shape[0]), int(g.shape[1])

                if ds is not None:
                    md, ok = safe_get_all_tags(ds)
                else:
                    md, ok = {}, False

                desc = safe_get(ds, "SeriesDescription") if ds is not None else ""
                has_contrast = False
                if ds is not None:
                    has_contrast = _truthy(safe_get(ds, "ContrastBolusAgent")) or _truthy(safe_get(ds, "ContrastBolusVolume"))
                ph = phase_label(desc or "", has_contrast)

                row = {
                    "tar_name": tar_name,
                    "dcm_name": dcm_name,
                    "slice_index": None,
                    "phase": ph,
                    "save_name": None,
                    "slice_status": "good"
                }
                if ok:
                    row.update(md)
                groups[(H, W)].append((g, row))

    except Exception as e:
        errfile = os.path.join(ERROR_DIR, f"{stem}_error.txt")
        with open(errfile, "a") as f:
            f.write(f"{datetime.datetime.now()} TAR-level exception: {repr(e)}\n{traceback.format_exc()}\n")
        log_global(f"[ERROR] TAR_READ_FAIL for {tar_name}: {repr(e)}")
        rel_name = f"{stem}/{stem}_1x1x1_{tar_name}.blosc"
        out_path = os.path.join(IMAGES_DIR, rel_name)
        WRITE_POOL.submit(write_blosc_int16, out_path, np.zeros((1, 1, 1), dtype=np.int16))
        row = {
            "tar_name": tar_name,
            "dcm_name": "TAR_READ_FAIL",
            "slice_index": 0,
            "phase": "UNKNOWN",
            "save_name": rel_name,
            "slice_status": "all_bad"
        }
        metadata_rows.append(row)
        return metadata_rows

    if not any_good:
        rel_name = f"{stem}/{stem}_1x1x1_{tar_name}.blosc"
        out_path = os.path.join(IMAGES_DIR, rel_name)
        WRITE_POOL.submit(write_blosc_int16, out_path, np.zeros((1, 1, 1), dtype=np.int16))
        errfile = os.path.join(ERROR_DIR, f"{stem}_error.txt")
        with open(errfile, "a") as f:
            f.write(f"{datetime.datetime.now()} ALL_SLICES_BAD: {tar_name} total_slices={total_slice_count}\n")
        row = {
            "tar_name": tar_name,
            "dcm_name": "ALL_SLICES_BAD",
            "slice_index": 0,
            "phase": "UNKNOWN",
            "save_name": rel_name,
            "slice_status": "all_bad"
        }
        metadata_rows.append(row)
        return metadata_rows

    try:
        for (H, W), lst in groups.items():
            if not lst:
                continue
            gs = [g for (g, _) in lst]
            vol = np.stack(gs, axis=-1)
            if np.issubdtype(vol.dtype, np.integer):
                vmin = vol.min()
                vmax = vol.max()
                if vmin >= -32768 and vmax <= 32767:
                    vol_i16 = vol.astype(np.int16, copy=False)
                else:
                    vol_i16 = to_int16_volume(vol.astype(np.float32, copy=False))
            else:
                vol_i16 = to_int16_volume(vol.astype(np.float32, copy=False))
            L = vol_i16.shape[-1]
            rel_name = f"{stem}/{stem}_{H}x{W}_{L}.blosc"
            out_path = os.path.join(IMAGES_DIR, rel_name)
            WRITE_POOL.submit(write_blosc_int16, out_path, vol_i16)
            for idx, (_, row) in enumerate(lst):
                row["save_name"] = rel_name
                row["slice_index"] = idx
                metadata_rows.append(row)
        return metadata_rows
    except Exception as e:
        errfile = os.path.join(ERROR_DIR, f"{stem}_error.txt")
        with open(errfile, "a") as f:
            f.write(f"{datetime.datetime.now()} POST_STACK_FAIL: {repr(e)}\n{traceback.format_exc()}\n")
        log_global(f"[ERROR] POST_STACK_FAIL for {tar_name}: {repr(e)}")
        rel_name = f"{stem}/{stem}_1x1x1_{tar_name}.blosc"
        out_path = os.path.join(IMAGES_DIR, rel_name)
        WRITE_POOL.submit(write_blosc_int16, out_path, np.zeros((1, 1, 1), dtype=np.int16))
        row = {
            "tar_name": tar_name,
            "dcm_name": "ALL_SLICES_BAD",
            "slice_index": 0,
            "phase": "UNKNOWN",
            "save_name": rel_name,
            "slice_status": "all_bad"
        }
        return [row]

# ================= Main =================
def main():
    '''
    Entry point for batch processing a subset of DICOM tar archives.

    The function:
      - Determines processing range from SUBSET_START / SUBSET_END.
      - Launches parallel workers to process each tar via process_one_tar().
      - Collects all slice-level metadata into a single DataFrame.
      - Writes metadata to a raw_<start>_<end>.csv file in TMP_META_DIR.
      - Logs progress and timing information throughout execution.

    Returns:
        None. All outputs are written to disk (Blosc volumes + CSV metadata).
    '''
    start = int(os.environ.get("SUBSET_START", "0"))
    end_env = os.environ.get("SUBSET_END")
    end   = None if end_env is None else int(end_env)

    all_tars = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tar.gz")))
    subset   = all_tars[start:end] if end is not None else all_tars
    total    = len(subset)
    task_id  = _task_id()
    log_global(f"START [{task_id}] subset {start}-{end if end is not None else 'END'} total={total}")

    if total == 0:
        log_global(f"WARN [{task_id}] no tars to process")
        return

    all_rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_one_tar, t): t for t in subset}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows = f.result()
            except Exception as e:
                errfile = os.path.join(ERROR_DIR, "future_error.txt")
                with open(errfile, "a") as ef:
                    ef.write(f"{datetime.datetime.now()} FUTURE_EXCEPTION: {repr(e)}\n{traceback.format_exc()}\n")
                log_global(f"[ERROR] FUTURE_EXCEPTION in main(): {repr(e)}")
                rows = None

            if rows:
                all_rows.extend(rows)
            if i % LOG_INTERVAL == 0 or i == total:
                elapsed = int(time.time() - t0)
                h, rem = divmod(elapsed, 3600)
                m, s = divmod(rem, 60)
                log_global(f"[{task_id}] progress {i}/{total} | {h}h{m}m{s}s")

    if all_rows:
        raw_prefix = os.path.join(TMP_META_DIR, f"raw_{start}_{end if end is not None else 'END'}")
        df_raw = pd.DataFrame(all_rows)
        front = ["tar_name", "dcm_name", "slice_index", "save_name", "phase"]
        others = [c for c in df_raw.columns if c not in front]
        df_raw = df_raw[front + others]
        safe_write_meta(df_raw, raw_prefix)
        log_global(f"[META] raw csv written: {raw_prefix}.csv (merge skipped)")

    elapsed = int(time.time() - t0)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    log_global(f"DONE [{task_id}] subset {start}-{end if end is not None else 'END'} | {h}h{m}m{s}s")

if __name__ == "__main__":
    main()