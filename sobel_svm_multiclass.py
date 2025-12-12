# Implementing out Sobel edge detection+CLAHE to better amplify the tortuosity of the blood vessels.
# extract Sobel edge features from retinal images,
# aggregate per patient/series, train an SVM to classify ROP vs Immature vs Other.

#Supervisor: Dr.Shyam Rajagopalan
#Date: July 10 2025

import os                        # filesystem ops
import sys                       # system interaction (exit)
import glob                      # pathname pattern matching
import re                        # regex for extracting digits from filenames
import shutil                    # file operations (remove dirs)
import argparse                  # CLI argument parsing
from tqdm import tqdm            # progress bars

import numpy as np               # numerical arrays
import pandas as pd              # tabular data handling
from PIL import Image            # save/load thumbnails
import cv2                       # image processing (OpenCV)
import matplotlib.pyplot as plt  # plotting (confusion matrix etc.)

# scikit-learn imports for model & evaluation
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib #save/load model & scaler


# image size we resize to before computing Sobel (keeps feature scale consistent)
image_size=224
# for reproducibility
rand_state=42
# output filenames
out_model="model.joblib"
out_scaler="scaler.joblib"
out_pred_csv= "test_predictions_sample_level.csv"

# folders the script will create for visual artifacts
misclass_dir= "misclassified" #thumbnails of misclassified sample groups
vis_dir="sample_visualizations" #small gallery of original + edge maps

# Which DG codes we consider ROP (change if your definition differs)
rop_codes= {2, 3, 4, 5, 6, 7, 8}   # *** EDITED: exclude 9 (status-post ROP) from active ROP set

# which code corresponds to "Immature retina" (ROP 0)
immature_code = 1                  # *** EDITED: code 1 corresponds to "ROP 0 / immature retina"

# Histogram bin count for Sobel magnitude histograms
sobel_bins= 32

# read the metadata from the csv
def read_csv_metadata(csv_path):
    """
    Read metadata CSV robustly (handles semicolon-delimited file).
    Returns: (df, id_col_name, diagnosis_col_name)
    """
    import pandas as pd
    # try reading with default separator first
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # fallback: let pandas infer separator using python engine
        df = pd.read_csv(csv_path, sep=None, engine='python')

    # if the CSV was parsed into a single column, likely the delimiter is ';'
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(csv_path, sep=';', engine='python')
            print("Re-read CSV using semicolon (';') delimiter.")
        except Exception:
            # last resort: let pandas infer with the python engine
            df = pd.read_csv(csv_path, sep=None, engine='python')

    # strip whitespace from column names for consistent access
    df.columns = [c.strip() for c in df.columns]

    # print a short preview so you can confirm the columns (helpful during debugging)
    print("CSV preview (first 5 rows):")
    print(df.head())
    print("Detected columns:",df.columns.tolist())

    # heuristics to pick the ID and diagnosis columns automatically
    id_col = None
    dg_col = None
    for c in df.columns:
        low = c.lower()
        if low in ("id", "i d", "sample_id", "sample id"):
            id_col = c
        if "diagnos" in low or "dg" in low or "diagn" in low:
            dg_col = c

    # fallback: choose a numeric column with many unique values as ID
    if id_col is None:
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
                if df[c].nunique() > 20:
                    id_col = c
                    break

    # fallback: find a column likely to contain codes/labels
    if dg_col is None:
        for c in df.columns:
            low = c.lower()
            if 'code' in low or 'dg' in low or 'diagn' in low:
                dg_col = c
                break

    print("Auto-detected ID column:", id_col, "Diagnosis column:", dg_col)
    if id_col is None or dg_col is None:
        # if auto-detection failed, print a message and return so the user can edit
        print("Couldn't reliably detect ID and/or diagnosis columns. Please inspect above and set them manually.")
        return df, id_col, dg_col

    # try to coerce the ID column to integers reliably
    try:
        # convert to numeric, allow coercion to capture possible dirty entries
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')
        if df[id_col].isna().any():
            n_miss = int(df[id_col].isna().sum())
            # if any row failed conversion, raise to force inspection
            raise ValueError(f"ID column '{id_col}' has {n_miss} non-numeric entries after coercion.")
        # safe: convert to plain int now
        df[id_col] = df[id_col].astype(int)
    except Exception as e:
        # provide a clear error message so you can debug the metadata
        raise RuntimeError(f"Failed to coerce ID column to int: {e}")

    return df, id_col, dg_col

# image collection from the folders
def folder_to_int(s):
    """
    Extract integer id from folder name or filename token.
    Returns integer (e.g., '001' -> 1) or None if not found.
    """
    if s is None:
        return None
    s = str(s)

    # if the entire token is digits, return as int
    if s.isdigit():
        try:
            return int(s)
        except:
            pass
    # using regex patterns to collect the images from the files
    # leading digits (e.g., '001_someinfo')
    m = re.match(r'^0*([0-9]+)', s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass

    # digits followed by underscore or hyphen (e.g.,'001-01' or '001_01')
    m = re.match(r'^0*([0-9]+)[_\-]', s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass

    # any contiguous digit sequence (first occurrence)
    m = re.search(r'([0-9]{1,6})', s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass

    # no match found
    return None

def collect_images(base_dir):
    """
    Collect images and infer sample_id.
    Supports:
      - directories where sample folders (001,002...) are direct children
      - single wrapper directory (auto-descend)
      - flat directory of images (infer sample id from filename)
      - recursive fallback
    Returns a DataFrame with columns: sample_id (string), image_path
    """
    base_dir = os.path.abspath(base_dir)
    rows = []

    # helper: try to collect assuming sample folders are direct children
    def collect_from_dir(d):
        r = []
        for sd in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isdir(sd):
                sample_id = os.path.basename(sd)
                found = False
                # collect common image extensions inside this sample folder
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.tif", "*.tiff"):
                    for p in glob.glob(os.path.join(sd, ext)):
                        r.append((sample_id, p))
                        found = True
                # if no images found directly, try one deeper (some datasets are nested)
                if not found:
                    for sub in sorted(glob.glob(os.path.join(sd, "*"))):
                        if os.path.isdir(sub):
                            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.tif", "*.tiff"):
                                for p in glob.glob(os.path.join(sub, ext)):
                                    # keep the top folder name as sample id
                                    r.append((sample_id, p))
        return r

    # first attempt: direct children
    rows = collect_from_dir(base_dir)

    # if nothing found and base_dir has exactly one subfolder, descend into it
    if not rows:
        children = [p for p in sorted(glob.glob(os.path.join(base_dir, "*"))) if os.path.isdir(p)]
        if len(children) == 1:
            print("Notice: base_dir contains a single subdirectory; descending to", children[0])
            rows = collect_from_dir(children[0])

    # if still empty, assume flat folder of images and infer sample id from filename
    if not rows:
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.tif", "*.tiff"):
            image_paths.extend(sorted(glob.glob(os.path.join(base_dir, ext))))
        if not image_paths:
            # recursive fallback in case files are nested further
            for p in glob.glob(os.path.join(base_dir, "**", "*.*"), recursive=True):
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    image_paths.append(p)
        for p in image_paths:
            fname = os.path.basename(p)
            name_wo_ext = os.path.splitext(fname)[0]
            sid_int = folder_to_int(name_wo_ext)
            if sid_int is not None:
                # store sample ids as zero-padded strings for consistency (001)
                sample_id = f"{sid_int:03d}"
            else:
                # fallback: use the filename (not ideal, but keeps the file in the dataset)
                sample_id = name_wo_ext
            rows.append((sample_id, p))

    df = pd.DataFrame(rows, columns=["sample_id", "image_path"])
    print(f"Collected {len(df)} images from {base_dir}")
    if len(df) > 0:
        # print a small preview to confirm inferred ids look sensible
        print("\nPreview (first 30 -> inferred sample_id):")
        for _, r in df.head(30).iterrows():
            print(os.path.basename(r['image_path']), "->", r['sample_id'])
    return df

# PREPROCESSING THE IMAGES
def preprocess_cv2(img_path, img_size=image_size, apply_clahe=True, circle_mask=True):
    """
    Read image with OpenCV, resize, optionally apply CLAHE and circular mask.
    Returns uint8 RGB image (not normalized) because Sobel expects intensity range.
    """
    # read image (BGR)
    img=cv2.imread(img_path)
    if img is None:
        raise RuntimeError("Failed to read: " + img_path)

    # convert to RGB (PIL/matplotlib friendly)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to fixed resolution — makes features consistent across images
    img=cv2.resize(img,(img_size, img_size), interpolation=cv2.INTER_AREA)

    # optional contrast enhancement: CLAHE on L channel in LAB color space
    if apply_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        img = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    # optional circular mask to remove black corners typical of fundus images
    if circle_mask:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), int(0.45 * min(h, w)), 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)

    return img  # uint8 RGB

# sobel feature extraction
def sobel_features_from_rgb(img_rgb, bins=sobel_bins):
    """
    Compute Sobel gradient magnitudes and return a compact feature vector:
      [mean_mag, std_mag, max_mag, normalized_histogram_of_magnitudes]
    """
    # convert to single-channel grayscale as Sobel operates on intensity
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Sobel derivatives in x and y (float32 for magnitude computation)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # gradient magnitude at each pixel
    mag = cv2.magnitude(gx, gy)

    # simple statistical summaries
    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))
    max_mag = float(np.max(mag))

    # histogram of magnitudes: gives distributional shape of edge strengths
    max_val = max(1.0, np.max(mag))  # avoid zero-range hist
    hist, _ = np.histogram(mag, bins=bins, range=(0.0, max_val))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-9)  # normalize to sum-to-one

    # final feature vector: a compact, interpretable descriptor
    feat = np.concatenate(([mean_mag, std_mag, max_mag], hist))
    return feat

# aggregation of features by sample
def aggregate_features_by_sample(df_with_feats):
    """
    Group image-level features by sample_id and return per-sample mean feature,
    along with the sample label and id_int for grouping.
    """
    agg = df_with_feats.groupby('sample_id').agg({
        'feat': lambda x: np.stack(x.values, axis=0).mean(axis=0),
        'label': 'first',   # label is same for all images in the sample
        'id_int': 'first'   # patient/series id (numeric) used for grouping
    }).reset_index()
    return agg

# saving the misclassified images for debugging
def save_misclassified_thumbnails(merged_images_df, sample_preds_df, out_dir=misclass_dir, thumb_size=256):
    """
    For each misclassified sample (sample-level), create a horizontal contact sheet of up to 4 images
    and save with filename indicating true label and predicted label.
    """
    # remove existing folder, create anew so outputs are deterministic
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # find misclassified rows
    mis = sample_preds_df[sample_preds_df['label'] != sample_preds_df['pred']]
    print("Saving thumbnails for", len(mis), "misclassified samples to", out_dir)

    # iterate with progress bar
    for _, row in tqdm(mis.iterrows(), total=len(mis)):
        sid = row['sample_id']
        true = row['label']
        pred = row['pred']

        # gather up to 4 images from that sample (some samples have many)
        imgs = merged_images_df[merged_images_df['sample_id'] == sid]['image_path'].tolist()
        n = min(4, len(imgs))
        thumbs = []
        for i in range(n):
            p = imgs[i]
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((thumb_size, thumb_size))
                thumbs.append(img)
            except Exception:
                # skip images that fail to load
                pass
        if not thumbs:
            continue

        # build a contact sheet horizontally
        total_w = sum(t.size[0] for t in thumbs)
        max_h = max(t.size[1] for t in thumbs)
        new_im = Image.new('RGB', (total_w, max_h), color=(0, 0, 0))
        x_offset = 0
        for t in thumbs:
            new_im.paste(t, (x_offset, 0))
            x_offset += t.size[0]

        # save with sample id + true/pred labels for easy inspection
        fname = f"{sid}_true-{true}_pred-{pred}.jpg"
        new_im.save(os.path.join(out_dir, fname))

# TRAINING AND EVALUATION
def train_and_evaluate(base_dir, csv_path, out_model=out_model, out_scaler=out_scaler):
    # read metadata CSV and detect columns
    print("Reading CSV...")
    df_meta, id_col, dg_col = read_csv_metadata(csv_path)
    if id_col is None or dg_col is None:
        print("ID or diagnosis column not found automatically. Edit the script to set id_col/dg_col.")
        return

    # collect images and inferred sample ids
    print("Collecting images...")
    df_imgs = collect_images(base_dir)

    # create numeric id_int column from sample_id strings (folder names or filename-derived)
    df_imgs['id_int'] = df_imgs['sample_id'].apply(folder_to_int)

    # report samples that failed conversion so you can debug naming issues
    n_missing = int(df_imgs['id_int'].isna().sum())
    if n_missing > 0:
        print(f"Warning: {n_missing} image rows have non-convertible sample id -> id_int. Examples:")
        print(df_imgs[df_imgs['id_int'].isna()]['sample_id'].unique()[:20])

    # drop any images we couldn't map to a numeric id (they can't be linked to metadata)
    df_imgs = df_imgs.dropna(subset=['id_int']).copy()
    df_imgs['id_int'] = df_imgs['id_int'].astype(int)

    # prepare a reduced metadata table: id_int + diagnosis code
    df_meta_small = df_meta[[id_col, dg_col]].drop_duplicates().rename(columns={id_col: "id_int", dg_col: "dg_code"})
    df_meta_small['id_int'] = pd.to_numeric(df_meta_small['id_int'], errors='coerce').astype(int)

    # merge image paths with metadata (left join keeps only images we had)
    merged = df_imgs.merge(df_meta_small, on='id_int', how='left')
    print("After merge, missing dg_code count:", int(merged['dg_code'].isna().sum()))

    # map diagnosis code (dg_code) to three classes: 'ROP', 'Immature', 'Other'
    def map_dg_to_three_classes(dg):
        """
        *** EDITED: New mapping for three classes:
            - 'Immature'  : dg == immature_code (1)  (ROP 0 / immature retina)
            - 'ROP'       : dg in rop_codes         (active ROP stages 1..5 and AP-ROP)
            - 'Other'     : everything else (includes physio 0, treated status 9, and other pathologies)
        """
        try:
            c = int(dg)
        except:
            return None
        if c == immature_code:
            return "Immature"           # *** EDITED
        elif c in rop_codes:
            return "ROP"                # *** EDITED
        else:
            return "Other"              # *** EDITED

    merged['label'] = merged['dg_code'].apply(map_dg_to_three_classes)

    # drop any images without a mapped label (shouldn't happen for well-formed CSV)
    merged = merged.dropna(subset=['label']).copy()
    print("Image-level label counts (may include duplicates per sample):")
    print(merged['label'].value_counts())

    # compute Sobel features per image (fast)
    feats = []
    for p in tqdm(merged['image_path'].tolist(), desc="computing Sobel features"):
        try:
            # preprocess returns uint8 RGB image
            img = preprocess_cv2(p, img_size=image_size, apply_clahe=True, circle_mask=True)
            feat = sobel_features_from_rgb(img, bins=sobel_bins)
            feats.append(feat)
        except Exception as e:
            # if a single image fails, we append None and report; later we drop rows with None
            feats.append(None)
            print("Error on", p, e)

    merged['feat'] = feats
    merged = merged[merged['feat'].notna()].copy()
    print("After dropping failed features:", len(merged), "images")

    # aggregate image-level features into one vector per sample by mean pooling
    agg = aggregate_features_by_sample(merged)

    # build arrays for modeling
    X = np.vstack(agg['feat'].values)     # feature matrix (samples x dims)
    y = agg['label'].values               # sample-level labels (three classes)
    groups = agg['id_int'].values         # patient ids for group splitting

    print("Sample-level shape:", X.shape)
    print("Label distribution (samples):")
    print(pd.Series(y).value_counts())

    # split by patient groups so no patient appears in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rand_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print("Train / Test samples:", X_train.shape[0], X_test.shape[0])

    # standardize features based on training data statistics only
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SVM baseline: balanced class weights to partially address imbalance
    svc = SVC(class_weight='balanced')
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale']}

    # stratified folds for hyperparameter search (stratify on y at sample-level)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand_state)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

    print("Starting GridSearchCV...")
    grid.fit(X_train_s, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # evaluate best model on test set
    y_pred = best.predict(X_test_s)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))   # handles multi-class
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    print("Confusion matrix:")
    print(cm)

    # save confusion matrix plot for README / report
    labels_unique = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix (sample-level)")
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

    # persist model + scaler so you can run prediction later
    joblib.dump(best, out_model)
    joblib.dump(scaler, out_scaler)
    print("Saved model ->", out_model, "scaler ->", out_scaler)

    # save sample-level predictions for inspection
    agg_test = agg.iloc[test_idx].copy()
    agg_test['pred'] = y_pred
    agg_test[['sample_id', 'id_int', 'label', 'pred']].to_csv(out_pred_csv, index=False)
    print("Saved sample-level predictions to",out_pred_csv)

    # save thumbnails for misclassified samples so you can show them in README
    save_misclassified_thumbnails(merged, agg_test, out_dir=misclass_dir)

    # save visualizations (original + Sobel magnitude) for a few test samples
    os.makedirs(vis_dir, exist_ok=True)
    for i, row in agg_test.head(12).iterrows():
        sid = row['sample_id']
        imgs = merged[merged['sample_id'] == sid]['image_path'].tolist()[:4]
        if not imgs:
            continue
        fig, axs = plt.subplots(2, len(imgs), figsize=(4 * len(imgs), 6))
        for j, p in enumerate(imgs):
            try:
                img = preprocess_cv2(p, img_size=image_size)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(gx, gy)
                mag_disp = (mag / (mag.max() + 1e-9))
                axs[0, j].imshow(img)
                axs[0, j].axis('off')
                axs[1, j].imshow(mag_disp, cmap='gray')
                axs[1, j].axis('off')
            except Exception as e:
                print("viz error", p, e)
        plt.suptitle(f"sample {sid} true={row['label']} pred={row['pred']}")
        plt.tight_layout()
        outfn = os.path.join(vis_dir, f"{sid}_viz.png")
        plt.savefig(outfn, dpi=150)
        plt.close(fig)
    print("Saved sample visualizations to",vis_dir)

    return best, scaler, agg

# Prediction on a sample
def predict_sample(model_path, scaler_path, sample_folder):
    """
    Load model and scaler, compute Sobel features of images in sample_folder,
    aggregate by mean and return predicted class and optional decision score.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # collect image files inside the sample folder
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.tif", "*.tiff"):
        imgs.extend(sorted(glob.glob(os.path.join(sample_folder, ext))))
    if not imgs:
        raise RuntimeError("No images in sample folder: " + sample_folder)

    # compute features for each image and average
    feats = []
    for p in imgs:
        img = preprocess_cv2(p, img_size=image_size)
        feat = sobel_features_from_rgb(img, bins=sobel_bins)
        feats.append(feat)
    feats = np.stack(feats, axis=0)
    feat_mean = feats.mean(axis=0, keepdims=True)

    # apply scaler and predict
    feat_s = scaler.transform(feat_mean)
    pred = model.predict(feat_s)[0]
    score = model.decision_function(feat_s) if hasattr(model, "decision_function") else None
    return pred, score

# for command line execution
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict"], required=True)
    p.add_argument("--base_dir", default=None, help="images base folder (e.g. /.../images/images)")
    p.add_argument("--csv", default=None, help="path to CSV metadata (required for train)")
    p.add_argument("--model", default=out_model, help="path to trained model.joblib")
    p.add_argument("--scaler", default=out_scaler, help="path to trained scaler.joblib")
    p.add_argument("--sample_folder", default=None, help="single sample folder for prediction")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        if not args.base_dir or not args.csv:
            print("Provide --base_dir and --csv for training.")
            sys.exit(1)
        train_and_evaluate(args.base_dir, args.csv)
    else:
        if not args.sample_folder:
            print("Provide --sample_folder for prediction.")
            sys.exit(1)
        pred, score = predict_sample(args.model, args.scaler, args.sample_folder)
        print("Prediction:", pred)
        if score is not None:
            print("Score:", score)
