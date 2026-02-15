# CE-VAE + TUDA: Deployment Guide

This guide explains exactly how to run your improved model on Kaggle or Google Colab.

## 1. Prepare Your Files

### Checkpoint Format

**Always use the `.ckpt` version** of the checkpoint. pyTorch Lightning (which this project uses) is designed to load directly from these files. You do **not** need to extract it.

### Code Transfer Options (GitHub Recommended)

I have created a dedicated repository for your project under your account:
**URL**: `https://github.com/priyanshuharshbodhi1/ce-vae-tuda-underwater-enhancement`

1. **GitHub (Best)**: Just run `!git clone https://github.com/priyanshuharshbodhi1/ce-vae-tuda-underwater-enhancement.git` in the notebook.
2. **Local ZIP**: If you prefer, zip your project folder (excluding `venv` and `data`) and upload it directly.

## 2. Option A: Running on Kaggle (Recommended)

### Step 1: Upload the Notebook from your PC

1. Go to your [Kaggle Home](https://www.kaggle.com/).
2. Click the **+ Create** button in the top left and select **New Notebook**.
3. Once the new notebook opens, go to **File** -> **Import Notebook**.
4. Click the **Upload** tab and select the `notebooks/train_cevae_tuda.ipynb` file from your computer.
5. In the right sidebar, under **Session Options**, set **Accelerator** to **GPU T4 x2**.

### Step 2: Clone the Repo

In the first cell of your Kaggle notebook, run:

```python
!git clone https://github.com/priyanshuharshbodhi1/ce-vae-tuda-underwater-enhancement.git
%cd ce-vae-tuda-underwater-enhancement
```

### Step 2: Add Datasets

1. In the right sidebar, click **+ Add Input**.
2. **Search** for:
   - `noureldin199/lsui-large-scale-underwater-image-dataset` (LSUI)
   - `larjeck/uieb-dataset-raw` (UIEB)
3. **Upload Checkpoint**: Click **+ Add Input** -> **Upload Dataset**. Upload your `lsui-cevae-epoch119.ckpt` file here. Kaggle will treat this file as a dataset.

### Step 3: Run

1. If you used GitHub, run the `!git clone` cell.
2. If you uploaded a ZIP, use the `!unzip` command.
3. Follow the notebook cells. It will take ~1.5 - 2 hours to train.
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
