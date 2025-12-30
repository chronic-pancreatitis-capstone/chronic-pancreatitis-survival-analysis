import os
import ast
import math
from datetime import datetime
import pylibjpeg
import numpy as np
import pandas as pd
import pydicom
import blosc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# BLOSC in-memory cache (per DataLoader worker)
BLOSC_CACHE = {}
BLOSC_CACHE_MAX = 32  

# Parameters
MAX_TOTAL_SLICES = 64     # max slices per patient (across all blocks)
MIN_PER_BLOCK    = 1       # minimum slices per block
TARGET_SIZE      = 256
META_DIM     = 2  

# Paths
ROOTS = [
    "<PRIVATE_DATA_PATH>",
    "<PRIVATE_DATA_PATH>",
    "<PRIVATE_DATA_PATH>",
]
ROOTS2 = [
    "<PRIVATE_DATA_PATH>",
]


# ============================================================================
# 1. Window parsing + slice loading (BLOSC + DICOM)
# ============================================================================
def parse_window_value(val):
    """
    Handle window values stored as scalar or '[c1, c2, ...]'.
    For multi-valued DICOM window center/width, we take the first element.
    """
    if val is None:
        raise ValueError("Window value is None")

    s = str(val).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return float(parsed[0])
        return float(parsed)
    return float(s)


def resolve_slice_path(row):
    """
    Decide whether this slice lives in BLOSC-serialized volume or raw DICOM.
    Returns:
        mode: "blosc" or "dicom"
        path: file path
    """
    save_name = row.get("save_name", "")
    save_name2 = row.get("save_name2", "")

    # First: save_name (blosc)
    if isinstance(save_name, str) and "/" in save_name:
        folder, filename = save_name.split("/", 1)
        for root in ROOTS:
            path = os.path.join(root, folder, filename)
            if os.path.isfile(path):
                return "blosc", path

    # Second: save_name2 (dicom)
    if isinstance(save_name2, str) and "/" in save_name2:
        folder, filename = save_name2.split("/", 1)
        for root in ROOTS2:
            path = os.path.join(root, folder, filename)
            if os.path.isfile(path):
                return "dicom", path

    raise FileNotFoundError(f"No valid file found:\n{row}")


def load_slice_from_image(path, wc, ww):
    """
    Load a single DICOM slice, apply HU conversion + window (wc, ww),
    safe for JPEG Lossless and JPEG2000 images.
    """
    dcm = pydicom.dcmread(path, force=True)

    # Pixel array / JPEG fallback
    try:
        raw = dcm.pixel_array
    except Exception as e1:
        print(f"[WARN] pixel_array failed for {path}: {e1}")
        from pydicom.pixel_data_handlers import pylibjpeg_handler as pj

        rows = int(getattr(dcm, "Rows", 0))
        cols = int(getattr(dcm, "Columns", 0))
        raw = pj.get_pixeldata(dcm)

        if rows and cols:
            raw = raw.reshape(rows, cols)

    # Always convert to float64 for HU math
    raw = raw.astype(np.float64)

    # HU conversion (most CT use slope, intercept)
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    hu = raw * slope + intercept

    # Apply DICOM window
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0

    img = np.clip(hu, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)
    return img


def load_slice_from_blosc(path, slice_idx, window_center, window_width):
    """
    Load a single slice from a BLOSC-compressed volume at given index,
    apply windowing, return np.float64 in [0,1].

    Uses a per-process in-memory cache to avoid repeatedly unpacking
    the same .blosc volume.
    """
    global BLOSC_CACHE, BLOSC_CACHE_MAX

    # Try cache first
    vol = BLOSC_CACHE.get(path)
    if vol is None:
        # Decompress once per worker
        with open(path, "rb") as f:
            packed = f.read()
        vol = blosc.unpack_array(packed)  # (H, W, Z) or (X, Y, Z)
        BLOSC_CACHE[path] = vol

        # Very simple eviction if cache too large
        if len(BLOSC_CACHE) > BLOSC_CACHE_MAX:
            # Pop an arbitrary key (you can make this smarter later)
            BLOSC_CACHE.pop(next(iter(BLOSC_CACHE)))

    # Now just slice
    if slice_idx < 0 or slice_idx >= vol.shape[2]:
        raise IndexError(
            f"slice_idx {slice_idx} out of range for volume with shape {vol.shape} ({path})"
        )

    img = vol[:, :, slice_idx].astype(np.float64)

    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower + 1e-6)

    return img



def load_slice(row):
    """
    High-level slice loader that:
      - resolves path (BLOSC vs DICOM)
      - parses window center / width
      - uses row_num as index
    """
    mode, path = resolve_slice_path(row)

    wc = parse_window_value(row["WindowCenter"])
    ww = parse_window_value(row["WindowWidth"])
    slice_idx = int(row["row_num"])   # ALWAYS use row_num

    if mode == "blosc":
        return load_slice_from_blosc(path, slice_idx, wc, ww)
    elif mode == "dicom":
        return load_slice_from_image(path, wc, ww)
    else:
        raise RuntimeError(f"Unexpected mode: {mode}")
    


# ============================================================================
# 2. Positional metadata (z_physical + acquisition time) – BLOCK-WISE
# ============================================================================
def parse_datetime_safe(date_str, time_str):
    """
    Convert DICOM date+time into seconds since epoch.
    Returns np.nan on failure.
    """
    if not isinstance(date_str, str) or not isinstance(time_str, str):
        return np.nan
    try:
        t_main = time_str.split(".")[0]  # drop fractional seconds
        dt = datetime.strptime(date_str + t_main, "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return np.nan


def _to_float_array(x, expected_len):
    """
    Helper to convert list/tuple/string → np.array(float)
    Returns None if parsing fails or length mismatch.
    """
    if x is None:
        return None

    if isinstance(x, str):
        x = x.strip()
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None

    try:
        arr = np.array(x, dtype=float)
    except Exception:
        return None

    if arr.shape[0] != expected_len:
        return None
    return arr


def build_slice_z_metadata(df_block):
    """
    Orientation-aware z-position → rank-normalized in [0,1] WITHIN THIS BLOCK.

    Returns:
        z_norm: torch.FloatTensor of shape (N,)
    """
    N = len(df_block)
    pos_vals = []

    for _, row in df_block.iterrows():
        s = None
        iop = _to_float_array(row["ImageOrientationPatient"], 6)
        ipp = _to_float_array(row["ImagePositionPatient"], 3)
        if iop is not None and ipp is not None:
            normal = np.cross(iop[:3], iop[3:])
            s = float(np.dot(normal, ipp))
        if s is None:
            # Fallback to scalar "z" if available
            s = float(row["z"])
        pos_vals.append(s)

    pos_vals = np.array(pos_vals, float)

    if N > 1:
        order = np.argsort(pos_vals)
        ranks = np.argsort(order).astype(float)
        z_norm = ranks / (N - 1)
    else:
        z_norm = np.zeros(N, dtype=float)

    return torch.tensor(z_norm, dtype=torch.float32)


# ============================================================================
# 3. Image preprocessing
# ============================================================================
def preprocess_ct_slice(ct_slice, target_size=TARGET_SIZE):
    """
    Convert numpy (H,W) → torch (1,3,target_size,target_size), scaled 0-1.
    """
    ct_slice = torch.as_tensor(ct_slice, dtype=torch.float32)
    ct_slice = ct_slice.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    ct_slice = F.interpolate(
        ct_slice,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    ct_slice = ct_slice.repeat(1, 3, 1, 1)  # (1,3,H,W)
    return ct_slice


# ============================================================================
# 4. Slice sampler, Dataset, dynamic collate
#    - Sample per block
#    - Then flatten all sampled blocks into ONE sequence per patient
# ============================================================================

class PatientSliceSampler:
    def __init__(
        self,
        max_total_slices=96,
        min_per_block=1,
        mode="train",                 # "train" | "val" | "test"
        global_seed=42,
        pancreas_labels=("PANCREAS",),
        region_col="body_region", # change if your column name differs
    ):
        self.max_total_slices = max_total_slices
        self.min_per_block = min_per_block
        self.mode = mode
        self.global_seed = global_seed
        self.pancreas_labels = tuple(x.upper() for x in pancreas_labels)
        self.region_col = region_col

    # RNG: random for train, deterministic for val/test (per patient id)
    def _make_rng(self, pid=None):
        if self.mode == "train" or pid is None:
            return np.random.default_rng()
        seed = (hash(str(pid) + str(self.global_seed)) & 0xFFFFFFFF)
        return np.random.default_rng(seed)

    def _pancreas_mask(self, df_block: pd.DataFrame):
        """
        Return bool mask where BodyRegion == PANCREAS.
        If no column or no PANCREAS slices, return None.
        """
        if self.region_col not in df_block.columns:
            return None

        regions = df_block[self.region_col].astype(str).str.upper()
        mask = regions.isin(self.pancreas_labels).to_numpy()
        return mask if mask.any() else None

    def __call__(self, block_dict, pid=None):
        """
        block_dict: { block_id → df_block }
        pid: patient id (for deterministic val/test sampling).
        """
        blocks = [(bid, len(df_block), df_block) for bid, df_block in block_dict.items()]
        if not blocks:
            return {}

        lengths = np.array([n for _, n, _ in blocks], dtype=int)
        total_len = lengths.sum()

        # under budget → keep all
        if total_len <= self.max_total_slices:
            return {bid: df_block for bid, _, df_block in blocks}

        # proportional quotas per block
        proportions = lengths / total_len
        quotas = (proportions * self.max_total_slices).astype(int)
        quotas = np.maximum(quotas, self.min_per_block)

        # enforce sum(quotas) <= max_total_slices
        while quotas.sum() > self.max_total_slices:
            idx = np.argmax(quotas)
            if quotas[idx] > self.min_per_block:
                quotas[idx] -= 1
            else:
                break

        rng = self._make_rng(pid)
        sampled_block_dict = {}

        for (bid, n, df_block), q in zip(blocks, quotas):
            # choose ordering columns
            if "z" in df_block.columns:
                sort_cols = ["z", "row_num"]
            elif "InstanceNumber" in df_block.columns:
                sort_cols = ["InstanceNumber", "row_num"]
            else:
                sort_cols = ["row_num"]

            df_block = df_block.sort_values(sort_cols).reset_index(drop=True)

            if q >= n:
                sampled_block = df_block
            else:
                pan_mask = self._pancreas_mask(df_block)  # bool array or None

                # divide along z into q bins (equal coverage)
                edges = np.linspace(0, n, q + 1, dtype=int)
                idxs = []

                for i in range(q):
                    start = edges[i]
                    end = max(edges[i + 1] - 1, edges[i])
                    cand = np.arange(start, end + 1)

                    if pan_mask is not None:
                        # block has PANCREAS slices → prefer them
                        pan_cand = cand[pan_mask[cand]]
                    else:
                        # no PANCREAS info in this block
                        pan_cand = np.array([], dtype=int)

                    if self.mode == "train":
                        # TRAIN: random within bin
                        if pan_cand.size > 0:
                            chosen = rng.choice(pan_cand)
                        else:
                            chosen = rng.integers(start, end + 1)
                    else:
                        # VAL/TEST: deterministic center within bin
                        if pan_cand.size > 0:
                            chosen = int(pan_cand[len(pan_cand) // 2])
                        else:
                            chosen = (start + end) // 2

                    idxs.append(chosen)

                idxs = np.sort(np.unique(idxs))
                if len(idxs) > q:
                    idxs = idxs[:q]

                sampled_block = df_block.iloc[idxs].reset_index(drop=True)

            # keep ordered by z/InstanceNumber
            sampled_block = sampled_block.sort_values(sort_cols).reset_index(drop=True)
            sampled_block_dict[bid] = sampled_block

        return sampled_block_dict


class PatientDataset(Dataset):
    def __init__(self, df, mode):
        self.slice_sampler = PatientSliceSampler(
            max_total_slices=MAX_TOTAL_SLICES,
            min_per_block=MIN_PER_BLOCK,
            mode=mode,
            global_seed=1234,
        )

        self.patients = df["pat_id"].unique()

        # Precompute per-patient block_dict
        self.patient_blocks = {}
        for pid in self.patients:
            df_p = df[df["pat_id"] == pid]

            block_dict = {}
            for (ser_uid, iop), df_block in df_p.groupby(
                ["SeriesInstanceUID", "ImageOrientationPatient"]
            ):
                block_id = f"{ser_uid}||{iop}"
                df_block = df_block.sort_values("z").reset_index(drop=True)
                block_dict[block_id] = df_block

            self.patient_blocks[pid] = block_dict

        # Survival labels per patient
        self.patient_labels = df.groupby("pat_id").agg({
            "time": "first",
            "event": "first"
        })

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        blocks = self.patient_blocks[pid]

        sampled = self.slice_sampler(blocks, pid=pid)

        imgs_all = []
        z_all = []
        time_all = []

        for bid in sorted(sampled.keys()):
            df_block = sampled[bid]

            # --- images ---
            imgs = []
            for _, row in df_block.iterrows():
                arr = load_slice(row)
                arr = preprocess_ct_slice(arr)
                imgs.append(arr)
            imgs = torch.cat(imgs, dim=0)  # (L,3,256,256)

            # --- z_norm in block ---
            z_norm = build_slice_z_metadata(df_block)  # (L,)

            # --- raw timestamps ---
            ts = []
            for _, row in df_block.iterrows():
                ts_val = parse_datetime_safe(
                    row.get("StudyDate", ""),
                    row.get("StudyTime", "")
                )
                ts.append(ts_val)

            imgs_all.append(imgs)
            z_all.append(z_norm)
            time_all.append(torch.tensor(ts, dtype=torch.float32))

        # flatten
        imgs = torch.cat(imgs_all, dim=0)      # (N,3,256,256)
        z = torch.cat(z_all, dim=0)            # (N,)
        t = torch.cat(time_all, dim=0)         # (N,)

        # normalize time within patient
        mask = torch.isfinite(t)
        count = mask.sum()

        if count > 0:
            t_mean = t[mask].mean()
        else:
            t_mean = torch.tensor(0.0, dtype=t.dtype)

        if count > 1:
            t_std = t[mask].std(unbiased=False)
        else:
            t_std = torch.tensor(1.0, dtype=t.dtype)

        if not torch.isfinite(t_std) or t_std < 1e-6:
            t_std = torch.tensor(1.0, dtype=t.dtype)

        t_norm = (t - t_mean) / t_std
        t_norm = torch.nan_to_num(t_norm)

        meta = torch.stack([z, t_norm], dim=1)  # (N,2)

        time = torch.tensor(self.patient_labels.loc[pid, "time"], dtype=torch.float32)
        event = torch.tensor(self.patient_labels.loc[pid, "event"], dtype=torch.float32)

        return {
            "imgs": imgs,
            "meta": meta,
            "time": time,
            "event": event,
            "pat_id": pid,
        }


# ============================================================================
# 5. Collate_FN
# ============================================================================
def dynamic_pad_collate(batch):
    """
    Flattened per-patient sequences:
      imgs: (L_i,3,256,256)
      meta: (L_i,2)

    Returns:
      imgs: (B,Lmax,3,256,256)
      meta: (B,Lmax,2)
      mask: (B,Lmax)  True = real slice
    """
    B = len(batch)
    Lmax = max(item["imgs"].shape[0] for item in batch)

    imgs_pad = torch.zeros(
        B, Lmax, 3, TARGET_SIZE, TARGET_SIZE, dtype=torch.float32
    )
    meta_pad = torch.zeros(B, Lmax, META_DIM, dtype=torch.float32)
    mask     = torch.zeros(B, Lmax, dtype=torch.bool)

    times, events, pat_ids = [], [], []

    for i, item in enumerate(batch):
        imgs = item["imgs"]
        meta = item["meta"]
        L    = imgs.shape[0]

        imgs_pad[i, :L] = imgs
        meta_pad[i, :L] = meta
        mask[i, :L]     = True

        times.append(item["time"])
        events.append(item["event"])
        pat_ids.append(item["pat_id"])

    return {
        "imgs": imgs_pad,
        "meta": meta_pad,
        "mask": mask,
        "time": torch.stack(times),
        "event": torch.stack(events),
        "pat_id": pat_ids,
    }

    


