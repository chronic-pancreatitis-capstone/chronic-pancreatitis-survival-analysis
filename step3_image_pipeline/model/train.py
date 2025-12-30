import os
import math
import sys
import numpy as np
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append("SAM_Med2D")
from sam_med2d_encoder import build_sam_med2d_encoder

from load_data import load_and_split
from preprocessing import PatientDataset, dynamic_pad_collate, TARGET_SIZE
from model import CoxModel, cox_ph_loss


# ============================================================================
# 0. CONFIG / HYPERPARAMETERS + DEVICE
# ============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE    = 16
ACC_STEPS     = 2          # effective batch = BATCH_SIZE * ACC_STEPS
NUM_WORKERS   = 5
PREFETCH      = 2
PERSISTENT    = False

NUM_EPOCHS    = 100
LR            = 2e-5
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
MIN_LR        = 1e-5

HORIZON_YEARS_LIST = [5.0, 10.0]
PRIMARY_HZ = 5.0

SAM_CKPT = "<PRIVATE_DATA_PATH>"


# ============================================================================
# AMP context (safe on CPU)
# ============================================================================

USE_CUDA = torch.cuda.is_available()
BF16_OK = USE_CUDA and torch.cuda.is_bf16_supported()

amp_ctx = (
    torch.cuda.amp.autocast(dtype=torch.bfloat16 if BF16_OK else torch.float16)
    if USE_CUDA else nullcontext()
)
scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)


# ============================================================================
# 1. Load dataframes & build datasets/dataloaders
# ============================================================================
train_df, val_df, test_df = load_and_split()
train_ds = PatientDataset(train_df, mode="train")
val_ds   = PatientDataset(val_df, mode="val")
test_ds  = PatientDataset(test_df, mode="test")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_CUDA,
    prefetch_factor=PREFETCH,
    persistent_workers=PERSISTENT,
    collate_fn=dynamic_pad_collate,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_CUDA,
    prefetch_factor=PREFETCH,
    persistent_workers=PERSISTENT,
    collate_fn=dynamic_pad_collate,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_CUDA,
    prefetch_factor=PREFETCH,
    persistent_workers=PERSISTENT,
    collate_fn=dynamic_pad_collate,
)

print(
    "n_train_batch:", len(train_loader),
    "n_val_batch:", len(val_loader),
    "n_test_batch:", len(test_loader),
)


# ============================================================================
# 2. Build SAM-Med2D encoder (FREEZE to avoid OOM)
# ============================================================================
ENCODER = build_sam_med2d_encoder(
    checkpoint=SAM_CKPT,
    model_type="vit_b",
    image_size=TARGET_SIZE,
    encoder_adapter=True,
    device=DEVICE,
)

for p in ENCODER.parameters():
    p.requires_grad = False
print("[INFO] SAM-Med2D encoder frozen")


# ============================================================================
# 3. Risk AUC at horizon
# ============================================================================

def risk_auc_at_horizon_days(model, loader, horizon_years, device=None, verbose=False):
    """
    AUC at a fixed horizon:
      - event before horizon -> positive
      - non-event after horizon OR event after horizon -> negative (event flipped to 0)
      - censored before horizon are excluded
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    T_days = horizon_years * 365.25

    all_risk, all_time, all_event = [], [], []
    dl_iter = tqdm(loader, desc=f"Eval {horizon_years}y", ncols=90) if verbose else loader

    with torch.no_grad():
        for batch in dl_iter:
            imgs  = batch["imgs"].to(device, non_blocking=USE_CUDA)
            meta  = batch["meta"].to(device, non_blocking=USE_CUDA)
            mask  = batch["mask"].to(device, non_blocking=USE_CUDA)
            time  = batch["time"].to(device, non_blocking=USE_CUDA)
            event = batch["event"].to(device, non_blocking=USE_CUDA)

            with amp_ctx:
                risk = model(imgs, meta, mask)

            all_risk.append(risk.detach().float().cpu())
            all_time.append(time.detach().float().cpu())
            all_event.append(event.detach().float().cpu())

    if not all_risk:
        return np.nan

    all_risk  = torch.cat(all_risk).numpy()
    all_time  = torch.cat(all_time).numpy()
    all_event = torch.cat(all_event).numpy().astype(int)

    time_days = all_time
    true_pos_mask = (all_event == 1) & (time_days <= T_days)
    true_neg_mask = ((all_event == 0) & (time_days >= T_days)) | ((all_event == 1) & (time_days > T_days))
    include_mask  = true_pos_mask | true_neg_mask

    # flip late events to 0 for fixed-horizon AUC
    all_event[(all_event == 1) & (time_days > T_days)] = 0

    y_true = all_event[include_mask]
    scores = all_risk[include_mask]

    if verbose:
        print(
            f"[{horizon_years}y] n_total={len(all_event)}, "
            f"n_used={include_mask.sum()}, "
            f"pos={y_true.sum()}, neg={len(y_true) - y_true.sum()}"
        )

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, scores)


# ============================================================================
# 4. Scheduler: Warmup + Cosine LR
# ============================================================================

def build_warmup_cosine_scheduler(optimizer, num_epochs, warmup_epochs, eta_min, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = eta_min / base_lr
        return min_ratio + (1.0 - min_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 5. Training loop with AMP + gradient accumulation (NO DataParallel)
# ============================================================================
model = CoxModel(encoder=ENCODER).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = build_warmup_cosine_scheduler(
    optimizer,
    num_epochs=NUM_EPOCHS,
    warmup_epochs=WARMUP_EPOCHS,
    eta_min=MIN_LR,
    base_lr=LR,
)

CHECKPOINT_DIR = "<PRIVATE_DATA_PATH>"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_auc_primary = -float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch} / {NUM_EPOCHS} =====")
    scheduler.step(epoch-1)
    print(f"LR at epoch {epoch}: {scheduler.get_last_lr()[0]:.2e}")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_loss_sum = 0.0
    n_steps = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}", ncols=90)):
        imgs  = batch["imgs"].to(DEVICE, non_blocking=USE_CUDA)
        meta  = batch["meta"].to(DEVICE, non_blocking=USE_CUDA)
        mask  = batch["mask"].to(DEVICE, non_blocking=USE_CUDA)
        time  = batch["time"].to(DEVICE, non_blocking=USE_CUDA)
        event = batch["event"].to(DEVICE, non_blocking=USE_CUDA)

        with amp_ctx:
            risk = model(imgs, meta, mask)
            loss = cox_ph_loss(risk, time, event) / ACC_STEPS

        if USE_CUDA:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % ACC_STEPS == 0:
            if USE_CUDA:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if USE_CUDA:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss_sum += loss.item() * ACC_STEPS
        n_steps += 1

    # leftover grads
    if (step + 1) % ACC_STEPS != 0:
        if USE_CUDA:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if USE_CUDA:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    train_loss = train_loss_sum / max(n_steps, 1)
    print(f"Train loss: {train_loss:.4f}")

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
    epoch_auc = {}
    with torch.no_grad():
        for hz in HORIZON_YEARS_LIST:
            auc_hz = risk_auc_at_horizon_days(
                model, val_loader, horizon_years=hz, device=DEVICE, verbose=True
            )
            epoch_auc[hz] = auc_hz
            print(f"{hz:.1f}-Year AUC: {auc_hz:.4f}")

    auc_primary = epoch_auc.get(PRIMARY_HZ, float("nan"))

    # save last
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "epoch_auc": epoch_auc,
            "primary_hz": PRIMARY_HZ,
            "best_auc_primary": best_auc_primary,
        },
        os.path.join(CHECKPOINT_DIR, "cox_sam_last.pth"),
    )

    # save best
    if not math.isnan(auc_primary) and auc_primary > best_auc_primary:
        best_auc_primary = auc_primary
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_auc_primary": best_auc_primary,
                "primary_hz": PRIMARY_HZ,
            },
            os.path.join(CHECKPOINT_DIR, f"cox_sam_best_{PRIMARY_HZ:.1f}y.pth"),
        )
        print(f"[CKPT] New best {PRIMARY_HZ:.1f}y AUC={best_auc_primary:.4f}")


# ============================================================================
# 6. FINAL TEST EVALUATION (best checkpoint)
# ============================================================================
print("\n===== FINAL TEST EVALUATION (best checkpoint) =====")
ckpt_best_path = os.path.join(CHECKPOINT_DIR, f"cox_sam_best_{PRIMARY_HZ:.1f}y.pth")

if not os.path.isfile(ckpt_best_path):
    print(f"[TEST] No best checkpoint found at {ckpt_best_path}. Skipping.")
else:
    print(f"[TEST] Loading best checkpoint from {ckpt_best_path}")
    checkpoint = torch.load(ckpt_best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        for hz in HORIZON_YEARS_LIST:
            auc_hz = risk_auc_at_horizon_days(
                model, test_loader, horizon_years=hz, device=DEVICE, verbose=True
            )
            print(f"[TEST] {hz:.1f}-Year AUC: {auc_hz:.4f}")


           
