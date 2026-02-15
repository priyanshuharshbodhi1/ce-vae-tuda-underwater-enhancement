# CE-VAE + TUDA: Deployment Guide

This guide explains exactly how to run your improved model on Kaggle or Google Colab.

## 1. Prepare Your Files

You need a single folder containing your code. If you are on your local machine, zip the entire project (excluding `venv`, `__pycache__`, and `data`):

```bash
zip -r ce-vae-tuda.zip . -x "venv/*" -x "__pycache__/*" -x "data/*" -x ".git/*"
```

Or simply upload the files manually.

## 2. Option A: Running on Kaggle (Recommended)

Kaggle gives you 30 hours of free T4 GPU per week and fast dataset access.

### Step 1: Create a New Notebook

1. Go to [Kaggle](https://www.kaggle.com/).
2. Click **Create** -> **New Notebook**.
3. In the right sidebar, under **Session Options**, set **Accelerator** to **GPU T4 x2** (or x1).

### Step 2: Upload Datasets

1. In the right sidebar, click **Add Input**.
2. **Search** for "LSUI Dataset" or upload your own `LSUI` folder.
3. **Search** for "UIEB Dataset" (for real images) or upload your own `real_underwater` folder.
4. **Upload** your pre-trained checkpoint `lsui-cevae-epoch119.ckpt` as a dataset.

### Step 3: Upload Code

1. Click **File** -> **Import Notebook** -> **Upload**.
2. Select `notebooks/train_cevae_tuda.ipynb`.
3. Kaggle might not let you upload the whole generic code folder easily. **Simpler Alternative:**
   - Create a generic "Script" or "Utility Script" on Kaggle with your `.py` files.
   - **OR (Easiest)**: Just clone your GitHub repo in the first cell of the notebook (as set up in the notebook).

### Step 4: Run

1. Run the cells in order.
2. The notebook handles dependencies (`albumentations`, `lightning`, etc.).
3. It will train for 30 epochs (~1.5 hours).
4. **Results**: Upon completion, check the **Output** section for `results_comparison.txt` and `comparison_results.png`.

---

## 3. Option B: Running on Google Colab (Easiest Upload)

Colab is great because you can drag-and-drop files easily.

### Step 1: Open Notebook

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** -> select `notebooks/train_cevae_tuda.ipynb`.

### Step 2: Enable GPU

1. Go to **Runtime** -> **Change runtime type**.
2. Select **T4 GPU**.

### Step 3: Upload Files

1. Click the file icon 📁 on the left sidebar.
2. Drag and drop your **entire code folder** (the unzipped contents) into the files area.
   - Specifically, ensure `src/`, `configs/`, and `main.py` are in the root `/content/` directory.
3. Upload your datasets to Google Drive (recommended for speed) and mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Copy data to local workspace for speed
   !cp -r /content/drive/MyDrive/LSUI ./data/
   ```

### Step 4: Run

1. Execute the notebook cells.
2. The training will effectively "fine-tune" your model.
3. **Download Results**: When finished, download `results_comparison.txt` and `cevae_tuda_finetuned.ckpt` from the file browser.

## 4. Tips for 3-4 Hour Completion

The config is already optimized for this:

- **30 Epochs**: Short enough to finish in <2 hours but long enough to learn domain features.
- **Mixed Precision**: Enabled (`precision: 16-mixed`) for 2x speed.
- **Frozen Encoder**: We freeze the early layers, so we update fewer parameters (faster).

## 5. Where are the scores?

At the end of the notebook execution, look for:

1. **Console Output**: A table comparing "Baseline CE-VAE" vs "CE-VAE + TUDA".
2. **File**: `results_comparison.txt` (saved in the same directory).
3. **Image**: `comparison_results.png` showing visual side-by-side improvements.
