import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split


def prepare_cox_data(df):
    """
    Prepare time and event variables for Cox regression.

    Final convention:
    - time is ALWAYS in **days** for BOTH events and censored.
    """

    data = df.copy()

    # event indicator is directly cp_label
    data["event"] = data["cp_label"].astype(float)

    # 1) Start with time_of_observation (already in days)
    data["time"] = data["time_of_observation"].astype(float)

    return data[["pat_id", "time", "event", "time_window", "cp_date", "first_ap","last_ap_date"]]

def extract_z(s):
    """Extract z from ImagePositionPatient string."""
    if not isinstance(s, str):
        return np.nan

    v = s.strip()
    if not v:
        return np.nan

    # Case 1: DICOM style "x\y\z"
    if "\\" in v:
        parts = v.split("\\")
        if len(parts) >= 3:
            try:
                return float(parts[2])
            except ValueError:
                return np.nan

    # Case 2: Python-list style "[-173.6, -312.6, -140.3]"
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
            return float(parsed[2])
    except Exception:
        pass

    # Fallback: maybe it's just a single float as string
    try:
        return float(v)
    except ValueError:
        return np.nan
    
def drop_unqualified_slices(df):
    df = df.copy()

    # Parse dates (all YYYYMMDD expected)
    df["study_dt"] = pd.to_datetime(df["StudyDate"].astype(str), format="%Y%m%d", errors="coerce")
    df["ap_dt"]    = pd.to_datetime(df["first_ap"], errors="coerce")
    df["cp_dt"]    = pd.to_datetime(df["cp_date"], errors="coerce")
    df["last_ap_dt"] = pd.to_datetime(df["last_ap_date"], errors="coerce")

    def is_valid(row):
        
        # ------ 2. Patients with CP ------
        if row["event"] == 1:
            # keep slices BEFORE CP
            if row["study_dt"] >= row["cp_dt"]:
                return False
            return True

        # ------ 3. Patients WITHOUT CP ------
        # keep slices before last AP
        if row["study_dt"] > row["last_ap_dt"]:
            return False

        return True   # keep everything before last AP (including pre-AP)

    mask = df.apply(is_valid, axis=1)
    return df[mask].reset_index(drop=True)

def load_and_split():
    """
    - Load ct_relevant_slices_merged_with_save_name2.csv
    - Load output.csv and build Cox labels
    - Merge by pat_id
    - Compute z, drop NA window
    - Anti-leak slice filtering
    - Patient-level stratified split → train/val/test slice-level dfs
    """
    # Load merged CSV
    df = pd.read_csv("<PRIVATE_DATA_PATH>", dtype=str,keep_default_na=True)  
    print("Loaded:", len(df), "rows")

    # Load output.csv  
    clinical = pd.read_csv("<PRIVATE_DATA_PATH>")
    label = clinical[["pat_id","cp_label","time_of_observation","time_to_event_years","time_window","cp_date","first_ap","last_ap_date"]]

    # Build Cox DF
    cox_df = prepare_cox_data(label)
    cox_df = cox_df[~((cox_df['time'] == 0) & (cox_df['event'] == 0))]

    # Merge by par_id and Compute z
    training_image_clean = df.merge(cox_df,on = "pat_id", how = "left")
    training_image_clean["z"] = training_image_clean["ImagePositionPatient"].apply(extract_z)
    training_image_clean = training_image_clean.dropna(subset=["WindowCenter", "WindowWidth"])

    # Drop unqualified slices
    training_image_clean_noleak = drop_unqualified_slices(training_image_clean)

    # Stratified train/val split (60% train, 20% val, 20% test)
    df_pat = training_image_clean_noleak.groupby("pat_id").agg({
    "time": "first",
    "event": "first"
    }).reset_index()

    df_pat["time_bin"] = pd.qcut(df_pat["time"], q=3, labels=False, duplicates="drop")
    df_pat["strata"]   = df_pat["event"].astype(str) + "_" + df_pat["time_bin"].astype(str)

    train_pat, temp_pat = train_test_split(
    df_pat["pat_id"],
    test_size=0.4,          
    random_state=42,
    stratify=df_pat["strata"],)


    df_temp = df_pat[df_pat["pat_id"].isin(temp_pat)].reset_index(drop=True)

    val_pat, test_pat = train_test_split(
    df_temp["pat_id"],
    test_size=0.5,           
    random_state=42,
    stratify=df_temp["strata"],
    )

    train_pat = set(train_pat)
    val_pat   = set(val_pat)
    test_pat  = set(test_pat)

    train_df = training_image_clean_noleak[training_image_clean_noleak["pat_id"].isin(train_pat)]
    val_df = training_image_clean_noleak[training_image_clean_noleak["pat_id"].isin(val_pat)]
    test_df = training_image_clean_noleak[training_image_clean_noleak["pat_id"].isin(test_pat)]

    print("n_train_pat:", len(train_pat),
          "n_val_pat:", len(val_pat),
          "n_test_pat:", len(test_pat),)
    
    return train_df, val_df, test_df


