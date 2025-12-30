# Because the original code is unavailable, here's a recreated version. Please verify its correctness.
"""
Extract patient-level image embeddings from a trained CoxModel and save to a .pt file.

This project expects these files in the same directory:
  - model.py            (CoxModel supports embedMode=True)
  - load_data.py        (load_and_split returns train_df, val_df, test_df)
  - preprocessing.py    (PatientDataset, dynamic_pad_collate, TARGET_SIZE)

And the SAM-Med2D code in:
  - SAM_Med2D/sam_med2d_encoder.py  (build_sam_med2d_encoder)

Default output file name:
  image_patient_embeddings.pt

Example:
  python save_feature.py \
    --ckpt cox_sam_checkpoints/cox_sam_best_5.0y.pth \
    --split test \
    --out image_patient_embeddings.pt
"""

import os
import sys
import argparse
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# match train.py import style
sys.path.append("SAM_Med2D")
from sam_med2d_encoder import build_sam_med2d_encoder  

from load_data import load_and_split  
from preprocessing import PatientDataset, dynamic_pad_collate, TARGET_SIZE  
from model import CoxModel  


def build_loader(df, mode, batch_size, num_workers, prefetch_factor, persistent_workers, use_cuda):
    """
    Build DataLoader consistent with preprocessing.py:
      - dynamic_pad_collate provides keys: imgs/meta/mask/time/event/pat_id
    """
    ds = PatientDataset(df, mode=mode)

    # DataLoader kwargs that depend on num_workers
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: stable ordering for feature alignment
        num_workers=num_workers,
        pin_memory=use_cuda,
        collate_fn=dynamic_pad_collate,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(ds, **dl_kwargs)


def extract_embeddings(model, loader, device, use_amp=True, prefer_bf16=True):
    """
    Returns a dict:
      features: (N, 256) float32 CPU tensor
      time:     (N,)     float32 CPU tensor
      event:    (N,)     float32 CPU tensor
      pat_id:   list of length N (strings)
    """
    model.eval()

    use_cuda = (device.type == "cuda") and torch.cuda.is_available()
    bf16_ok = use_cuda and prefer_bf16 and torch.cuda.is_bf16_supported()

    amp_ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16 if bf16_ok else torch.float16)
        if (use_cuda and use_amp)
        else nullcontext()
    )

    feats_all = []
    time_all = []
    event_all = []
    pat_id_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract embeddings", ncols=90):
            imgs = batch["imgs"].to(device, non_blocking=use_cuda)
            meta = batch["meta"].to(device, non_blocking=use_cuda)
            mask = batch["mask"].to(device, non_blocking=use_cuda)

            # In your collate_fn, time/event are tensors of shape (B,)
            time = batch["time"].detach().cpu()
            event = batch["event"].detach().cpu()

            # pat_id is a python list of length B in your collate_fn
            pat_id = batch["pat_id"]

            with amp_ctx:
                # embedMode=True => model returns patient_vec (B, 256)
                patient_vec = model(imgs, meta, mask)

            feats_all.append(patient_vec.detach().float().cpu())
            time_all.append(time)
            event_all.append(event)
            pat_id_all.extend(list(pat_id))

    features = torch.cat(feats_all, dim=0) if len(feats_all) else torch.empty((0, 256), dtype=torch.float32)
    time = torch.cat(time_all, dim=0) if len(time_all) else torch.empty((0,), dtype=torch.float32)
    event = torch.cat(event_all, dim=0) if len(event_all) else torch.empty((0,), dtype=torch.float32)

    return {
        "features": features,
        "time": time,
        "event": event,
        "pat_id": pat_id_all,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Best checkpoint .pth containing key 'model_state'.")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"],
                    help="Which split to extract embeddings for.")
    ap.add_argument("--out", type=str, default="<PRIVATE_DATA_PATH>",
                    help="Output .pt path (used if --split != all).")
    ap.add_argument("--out_dir", type=str, default="embeddings_pt",
                    help="Output directory (used if --split=all).")

    # match train.py defaults
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=5)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--persistent", action="store_true", help="Enable persistent workers.")

    ap.add_argument("--sam_ckpt", type=str, default="<PRIVATE_DATA_PATH>",
                    help="SAM-Med2D checkpoint path.")
    ap.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP during extraction.")

    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")
    print(f"[INFO] Device: {device}")

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # 1) Load data splits
    train_df, val_df, test_df = load_and_split()

    # 2) Build frozen SAM encoder (same as train.py)
    encoder = build_sam_med2d_encoder(
        checkpoint=args.sam_ckpt,
        model_type="vit_b",
        image_size=TARGET_SIZE,
        encoder_adapter=True,
        device=device,
    )
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print("[INFO] SAM encoder frozen + eval")

    # 3) Build CoxModel in embedMode
    model = CoxModel(encoder=encoder, embedMode=True).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model_state" not in ckpt:
        raise KeyError("Checkpoint missing key 'model_state'. Pass the .pth saved by train.py (best/last).")

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print(f"[INFO] Loaded model_state from: {args.ckpt}")

    # 4) Build loaders
    loaders = {}
    if args.split in ["train", "all"]:
        loaders["train"] = build_loader(
            train_df, mode="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch,
            persistent_workers=args.persistent,
            use_cuda=use_cuda,
        )
    if args.split in ["val", "all"]:
        loaders["val"] = build_loader(
            val_df, mode="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch,
            persistent_workers=args.persistent,
            use_cuda=use_cuda,
        )
    if args.split in ["test", "all"]:
        loaders["test"] = build_loader(
            test_df, mode="test",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch,
            persistent_workers=args.persistent,
            use_cuda=use_cuda,
        )

    # 5) Extract + save
    if args.split == "all":
        os.makedirs(args.out_dir, exist_ok=True)
        for split_name, loader in loaders.items():
            out_path = os.path.join(args.out_dir, f"{split_name}_image_patient_embeddings.pt")
            obj = extract_embeddings(
                model=model,
                loader=loader,
                device=device,
                use_amp=(not args.no_amp),
            )
            torch.save(obj, out_path)
            print(f"[OK] Saved {split_name}: features={tuple(obj['features'].shape)} -> {out_path}")
    else:
        split_name = args.split
        out_path = args.out
        obj = extract_embeddings(
            model=model,
            loader=loaders[split_name],
            device=device,
            use_amp=(not args.no_amp),
        )
        torch.save(obj, out_path)
        print(f"[OK] Saved {split_name}: features={tuple(obj['features'].shape)} -> {out_path}")


if __name__ == "__main__":
    main()
