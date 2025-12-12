#Implementing hog+svm for binary classification of RoP images
#Supervisor: Dr.Shyam Rajagopalan
#Date: Apr 25 2025
"""hog + color histogram -> svm for binary rop vs normal classification.

Usage:
  Train & evaluate:
    python rop_hog_svm.py-mode train --base_dir /path/to/images_stack_without_captions --csv /path/to/infant_retinal_database_info.csv

  Predict on a single sample (after training):
    python rop_hog_svm.py --mode predict --model model.joblib --scaler scaler.joblib --sample_folder /path/to/001

Outputs:
  - model.joblib, scaler.joblib
  - test_predictions_sample_level.csv
  - misclassified/  (thumbnails of misclassified samples with true_pred in filename)
  - report_confusion.png
"""

import os
import glob
import argparse
import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt


img_size = 224
batch_size = 32
random_state = 42

out_model = "model.joblib"
out_scaler = "scaler.joblib"
out_pred_csv = "test_predictions_sample_level.csv"
misclass_dir = "misclassified"

# diagnosis codes that we treat as active ROP
rop_codes = {2, 3, 4, 5, 6, 7, 8, 9}


def read_csv_metadata(csv_path):
    """
    load the metadata csv robustly (handles semicolon or comma).
    prints a preview and tries to auto-detect id & diagnosis columns.
    returns: dataframe, id_col_name, diagnosis_col_name
    """
    import pandas as pd
    # try a normal read first
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # fallback to python engine and let pandas guess delimiter
        df = pd.read_csv(csv_path, sep=None, engine='python')

    # if everything ended up in one column, try semicolon
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(csv_path, sep=';', engine='python')
            print("re-read csv with semicolon delimiter.")
        except Exception as e:
            print("failed semicolon parse:", e)
            raise

    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    print("csv preview (first 5 rows):")
    print(df.head())
    print("detected columns:", df.columns.tolist())

    # heuristics to find id and diagnosis columns
    id_col = None
    dg_col = None
    for c in df.columns:
        low = c.lower()
        if low in ("id", "i d", "sample_id", "sample id"):
            id_col = c
        if "diagnos" in low or "dg" in low or "diagn" in low:
            dg_col = c

    # fallback: pick a numeric column with many unique values as id
    if id_col is None:
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
                if df[c].nunique() > 20:
                    id_col = c
                    break

    # fallback for diagnosis-like column
    if dg_col is None:
        for c in df.columns:
            if 'code' in c.lower() or 'dg' in c.lower() or 'diagn' in c.lower():
                dg_col = c
                break

    print("auto-detected id column:", id_col, "diagnosis column:", dg_col)
    if id_col is None or dg_col is None:
        print("couldn't reliably detect id and/or diagnosis columns. inspect printed columns and set them manually if needed.")
        return df, id_col, dg_col

    # coerce id column safely to int
    try:
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')
        if df[id_col].isna().any():
            n_missing = df[id_col].isna().sum()
            raise ValueError(f"column {id_col} has {n_missing} non-numeric entries after coercion")
        df[id_col] = df[id_col].astype(int)
    except Exception as e:
        raise RuntimeError(f"failed to coerce id column to int: {e}")

    return df, id_col, dg_col

def collect_images(base_dir):
    """
    collect images assuming structure:
      base_dir/<sample_id>/*.jpg
    fallback: recursive search and attempt to infer sample_id from path components.
    returns dataframe with columns [sample_id, image_path]
    """
    rows = []
    for sd in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if os.path.isdir(sd):
            sample_id = os.path.basename(sd)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
                for p in glob.glob(os.path.join(sd, ext)):
                    rows.append((sample_id, p))

    # if nothing found with the simple pattern, do a recursive search
    if not rows:
        for p in glob.glob(os.path.join(base_dir, "**", "*.*"), recursive=True):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                parts = p.split(os.sep)
                # try to take folder two levels up as sample id, else one level
                sample_id = parts[-3] if len(parts) >= 3 else parts[-2]
                rows.append((sample_id, p))

    df = pd.DataFrame(rows, columns=["sample_id", "image_path"])
    print(f"collected {len(df)} images from {base_dir}")
    return df

def folder_to_int(s):
    """
    try to turn a folder-name (like '001' or '001_info') into an int id.
    returns integer or None if no digits found.
    """
    try:
        return int(s)
    except Exception:
        toks = str(s).split('_')
        for t in toks:
            if t.isdigit():
                return int(t)
        # fallback: extract digits anywhere
        digits = ''.join([ch for ch in str(s) if ch.isdigit()])
        return int(digits) if digits else None

def map_dg_to_binary(dg):
    """map diagnosis code to 'ROP' or 'Normal' based on rop_codes set"""
    try:
        c = int(dg)
    except Exception:
        return None
    return "ROP" if c in rop_codes else "Normal"

# preprocessing helpers
def preprocess_cv2(img_path, img_size=img_size, apply_clahe=True, circle_mask=True):
    """
    read with cv2, resize, optionally apply CLAHE and a circular mask to remove black corners.
    returns float32 image scaled [0..1].
    """
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("failed to read: " + img_path)

    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if apply_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        img = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    if circle_mask:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), int(0.45 * min(h, w)), 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)

    return img.astype(np.float32) / 255.0

def image_to_hog_color(img_rgb, hog_pixels_per_cell=(16, 16)):
    """
    compute hog features from grayscale + 3-channel color histograms, then concat.
    returns 1d numpy array.
    """
    # skimage hog expects 0..1 floats or uint8, we use floats in [0..1]
    gray = (cv2.cvtColor((img_rgb * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)).astype(np.float32) / 255.0
    hog_feats = hog(gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)
    hist = []
    for c in range(3):
        hist_c = cv2.calcHist([(img_rgb * 255).astype('uint8')], [c], None, [16], [0, 256]).flatten()
        hist_c = hist_c / (hist_c.sum() + 1e-9)
        hist.append(hist_c)
    return np.concatenate([hog_feats, np.concatenate(hist)])

def visualize_hog(img_rgb, hog_pixels_per_cell=(16, 16)):
    """
    return a rescaled hog image useful for plotting next to the original.
    """
    from skimage import exposure
    gray = (cv2.cvtColor((img_rgb * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)).astype(np.float32) / 255.0
    fd, hog_image = hog(gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                        visualize=True, feature_vector=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, np.max(hog_image)))
    return hog_image_rescaled

def aggregate_features_by_sample(df_with_feats):
    """
    group image-level features by sample_id, return per-sample mean feature vector.
    keeps sample_id, first label encountered, and id_int (numeric id) for grouping.
    """
    agg = df_with_feats.groupby('sample_id').agg({
        'feat': lambda x: np.stack(x.values, axis=0).mean(axis=0),
        'label': 'first',
        'id_int': 'first'
    }).reset_index()
    return agg

def save_misclassified_thumbnails(merged_images_df, sample_preds_df, out_dir=misclass_dir, thumb_size=256):
    """
    for each misclassified sample (sample-level), create a simple horizontal contact
    sheet with up to 4 thumbnails. filename includes true and predicted labels.
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mis = sample_preds_df[sample_preds_df['label'] != sample_preds_df['pred']]
    print("saving thumbnails for", len(mis), "misclassified samples to", out_dir)
    for _, row in tqdm(mis.iterrows(), total=len(mis)):
        sid = row['sample_id']
        true = row['label']
        pred = row['pred']
        imgs = merged_images_df[merged_images_df['sample_id'] == sid]['image_path'].tolist()
        n = min(4, len(imgs))
        thumbs = []
        for i in range(n):
            p = imgs[i]
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((thumb_size, thumb_size))
                thumbs.append(img)
            except Exception as e:
                print("error loading", p, e)
        if not thumbs:
            continue
        total_w = sum(t.size[0] for t in thumbs)
        max_h = max(t.size[1] for t in thumbs)
        new_im = Image.new('RGB', (total_w, max_h), color=(0, 0, 0))
        x_offset = 0
        for t in thumbs:
            new_im.paste(t, (x_offset, 0))
            x_offset += t.size[0]
        fname = f"{sid}_true-{true}_pred-{pred}.jpg"
        new_im.save(os.path.join(out_dir, fname))

def train_and_evaluate(base_dir, csv_path, out_model=out_model, out_scaler=out_scaler, debug=False):
    """
    main training pipeline:
      - read csv metadata
      - collect images and map to sample ids
      - merge with metadata to get labels
      - compute hog+color features per image
      - aggregate per sample, split groups, train svm with gridsearch
      - evaluate and save artifacts
    """
    print("reading csv...")
    df_meta, id_col, dg_col = read_csv_metadata(csv_path)

    print("collecting images...")
    df_imgs = collect_images(base_dir)

    # robustly convert sample folder names to numeric ids that match csv
    df_imgs['id_int'] = df_imgs['sample_id'].apply(folder_to_int)
    n_missing = int(df_imgs['id_int'].isna().sum())
    if n_missing > 0:
        print(f"warning: {n_missing} image rows have non-convertible sample_id -> id_int.")
        bad_samples = df_imgs[df_imgs['id_int'].isna()]['sample_id'].unique()
        print("example problematic sample folder names (up to 20):", bad_samples[:20])

    # drop images we cannot link to metadata
    df_imgs = df_imgs.dropna(subset=['id_int']).copy()
    df_imgs['id_int'] = df_imgs['id_int'].astype(int)

    # prepare small metadata table for merging
    df_meta_small = df_meta[[id_col, dg_col]].drop_duplicates().rename(columns={id_col: "id_int", dg_col: "dg_code"})
    df_meta_small['id_int'] = pd.to_numeric(df_meta_small['id_int'], errors='coerce').astype(int)

    # merge image paths with metadata on numeric id
    merged = df_imgs.merge(df_meta_small, on='id_int', how='left')
    print("after merge, missing dg_code count:", int(merged['dg_code'].isna().sum()))

    # map diagnosis codes to binary labels
    merged['label'] = merged['dg_code'].apply(map_dg_to_binary)
    print("image-level label counts (may include duplicates per sample):")
    print(merged['label'].value_counts(dropna=False))

    merged = merged.dropna(subset=['label']).copy()
    print("labeled images:", len(merged))

    # compute features per image
    feats = []
    for p in tqdm(merged['image_path'].tolist(), desc="computing HOG+color"):
        try:
            img = preprocess_cv2(p, img_size=img_size, apply_clahe=True, circle_mask=True)
            feats.append(image_to_hog_color(img))
        except Exception as e:
            feats.append(None)
            print("error", p, e)
    merged['feat'] = feats
    merged = merged[merged['feat'].notna()].copy()
    print("after dropping failed features:", len(merged), "images")

    # aggregate per sample
    agg = aggregate_features_by_sample(merged)
    X = np.vstack(agg['feat'].values)
    y = agg['label'].values
    groups = agg['id_int'].values
    print("sample-level shape:", X.shape)
    print("label distribution (samples):")
    print(pd.Series(y).value_counts())

    # group-wise train/test split (avoid patient leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print("train / test samples:", X_train.shape[0], X_test.shape[0])

    # scale features based on training stats only
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # svm baseline with grid search
    svc = SVC(class_weight='balanced')
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale']}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

    print("starting GridSearchCV...")
    grid.fit(X_train_s, y_train)
    best = grid.best_estimator_
    print("best params:", grid.best_params_)

    # evaluate on test set
    y_pred = best.predict(X_test_s)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("classification report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix:")
    print(cm)

    # save confusion matrix plot (sample-level)
    labels_unique = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("confusion matrix (sample-level)")
    ax.set_xticks(np.arange(len(labels_unique)))
    ax.set_yticks(np.arange(len(labels_unique)))
    ax.set_xticklabels(labels_unique)
    ax.set_yticklabels(labels_unique)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("report_confusion.png", dpi=150)
    plt.close(fig)

    # persist model and scaler
    joblib.dump(best, out_model)
    joblib.dump(scaler, out_scaler)
    print("saved model ->", out_model, "scaler ->", out_scaler)

    # save sample-level predictions for inspection
    agg_test = agg.iloc[test_idx].copy()
    agg_test['pred'] = y_pred
    agg_test[['sample_id', 'id_int', 'label', 'pred']].to_csv(out_pred_csv, index=False)
    print("saved sample-level predictions to", out_pred_csv)

    # save thumbnails for misclassified samples
    save_misclassified_thumbnails(merged, agg_test, out_dir=misclass_dir)

    # visualizations: show hog vs original for a few test samples
    viz_dir = "sample_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    for i, row in agg_test.head(6).iterrows():
        sid = row['sample_id']
        imgs = merged[merged['sample_id'] == sid]['image_path'].tolist()[:4]
        if not imgs:
            continue
        fig, axs = plt.subplots(2, min(4, len(imgs)), figsize=(4 * min(4, len(imgs)), 6))
        for j, p in enumerate(imgs):
            try:
                img = preprocess_cv2(p, img_size=img_size)
                hog_img = visualize_hog(img)
                axs[0, j].imshow((img * 255).astype('uint8'))
                axs[0, j].axis('off')
                axs[1, j].imshow(hog_img, cmap='gray')
                axs[1, j].axis('off')
            except Exception as e:
                print("viz error", p, e)
        plt.suptitle(f"sample {sid} true={row['label']} pred={row['pred']}")
        plt.tight_layout()
        outfn = os.path.join(viz_dir, f"{sid}_viz.png")
        plt.savefig(outfn, dpi=150)
        plt.close(fig)
    print("saved sample visualizations to", viz_dir)

    return best, scaler, agg

def predict_sample(model_path, scaler_path, sample_folder):
    """
    load a saved model + scaler and predict for a single sample folder.
    average image-level features to get a sample-level vector.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
        imgs.extend(sorted(glob.glob(os.path.join(sample_folder, ext))))
    if not imgs:
        raise RuntimeError("no images in sample folder: " + sample_folder)

    feats = []
    for p in imgs:
        img = preprocess_cv2(p, img_size=img_size)
        feats.append(image_to_hog_color(img))
    feats = np.stack(feats, axis=0)
    feat_mean = feats.mean(axis=0, keepdims=True)

    feat_s = scaler.transform(feat_mean)
    pred = model.predict(feat_s)[0]
    score = model.decision_function(feat_s) if hasattr(model, "decision_function") else None
    return pred, score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict"], required=True)
    p.add_argument("--base_dir", default=None, help="images base folder")
    p.add_argument("--csv", default=None, help="path to csv metadata")
    p.add_argument("--model", default=out_model)
    p.add_argument("--scaler", default=out_scaler)
    p.add_argument("--sample_folder", default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        if not args.base_dir or not args.csv:
            print("provide --base_dir and --csv for training.")
            raise SystemExit(1)
        train_and_evaluate(args.base_dir, args.csv)
    else:
        if not args.sample_folder:
            print("provide --sample_folder for prediction.")
            raise SystemExit(1)
        pred, score = predict_sample(args.model, args.scaler, args.sample_folder)
        print("prediction:", pred)
        if score is not None:
            print("score:", score)
