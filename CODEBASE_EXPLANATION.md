# CE-VAE Codebase: A Detailed & Simple Guide

This document explains exactly how this repository works, file by file and function by function, in simple terms. This is intended for your research project on **TUDA (Target-oriented Underwater Domain Adaptation)**.

---

## 🛠️ The Core Entry Points

### 1. `main.py` (The Training Engine)

This is the file you run to "teach" the model.

- **`SetupCallback`**: Sets up folders for logs and checkpoints so you don't lose progress.
- **`ImageLogger`**: Periodically saves pictures during training so you can see if the model is getting better or worse.
- **`main` block**:
  - Loads your configuration (what settings to use).
  - Initializes the model and data.
  - Scales the learning rate (how fast it learns) based on how many GPUs you have.
  - Starts the training loop.

### 2. `test.py` (The Tester / Inference)

This is the file you run to use a trained model on new images.

- **`DatasetFromFolder`**: A helper that grabs all images from a folder and gets them ready for the model.
- **`load_cevae`**: Loads the "brain" (model) and plugs in the "knowledge" (checkpoint weights).
- **`reconstruction_batch`**: The actual function that takes a batch of images and makes them look better.
- **`run`**: The main function that loops through your images, enhances them, and saves them to a new folder.

---

## 🧠 The "Brain" (Models & Architecture)

### 📂 `src/models/cevae.py`

This is the heart of the project. It defines the `CEVAE` class.

- **`__init__`**: Sets up the encoder, capsules, and decoder.
- **`encode`**: Takes an image and turns it into a "summarized" version (latent space). It uses both standard features and capsules.
- **`decode`**: Takes that summary and builds back a full-size, enhanced image.
- **`forward`**: The standard way data flows: Image → Encode → Decode → Enhanced Image.
- **`configure_optimizers`**: Tells the computer how to update the model's weights using math (Adam optimizer).

### 📂 `src/modules/capsules/` (Pattern Spotters)

Capsules are special because they understand **groups** of pixels as "objects" rather than just random dots.

- **`primary.py` (`PrimaryCaps`)**: Takes normal math features and groups them into "primary capsules."
- **`digit.py` (`DigitCaps`)**: Uses something called **Dynamic Routing**. It’s like a voting system where smaller capsules decide which larger "object" capsule they belong to.
- **`common.py` (`squash`)**: A simple math trick to make sure the capsule vectors stay a reasonable size.

### 📂 `src/modules/autoencoder/` (Builders)

- **`encoder.py`**: A series of shrinking steps that use "Attention" (focusing on important parts) to understand the image.
- **`decoder.py`**: A series of growing steps. It takes the capsule data and "skip connections" (original details) to rebuild the image.

---

## 📏 The "Regulators" (Losses & Metrics)

### 📂 `src/modules/losses/combined.py` (The Graders)

During training, these functions "grade" the model's work.

- **`ReconstructionLoss`**: Checks how similar the new image is to the original "perfect" one using:
  - **L1**: Pixel-to-pixel matching.
  - **Perceptual (LPIPS)**: Does it look realistic to a human eye?
  - **SSIM**: Does the structure look right?
  - **Color**: Are the colors natural for underwater?
- **`PatchGANDiscriminator`**: A "rival" model that tries to spot fakes. This forces our main model to create extremely realistic textures.

### 📂 `src/metrics/` (The Scorekeepers)

These files measure how good the final results are.

- **`__init__.py`**: The central place to calculate all scores.
- **`psnr` & `ssim`**: Standard ways to measure image quality.
- **`uiqm` & `uciqe`**: Special math formulas designed just for **underwater** images (measuring color, sharpness, and contrast).

---

## 📦 The "Gatherers" (Data Loading)

### 📂 `src/data/`

- **`image_enhancement.py`**: Reads "paired" lists (degraded image + ground truth image) from text files.
- **`base.py` (`ImagePairDatasetFromPaths`)**: The actual worker that:
  - Loads the image file from the disk.
  - **Resizes** it.
  - **Augments** it (randomly flips or changes colors to make the model smarter).
  - Normalizes the pixels to be between -1 and 1.

---

## 🛠️ Utilities & Build Tools

- **`src/util.py`**: Contains helper functions to download models from the internet (`download`) and check if they are correct (`md5_hash`).
- **`src/build/from_config.py`**: A "factory" function. It reads your `.yaml` settings file and automatically creates the right Python objects.

---

## 🔬 Advice for your TUDA Project

Since you are working on **Domain Adaptation** (transferring knowledge from one type of water to another):

1.  **Modify the Capsules**: The `DigitCaps` in `src/modules/capsules/digit.py` is where the "entity" knowledge lives. You might want to adapt these vectors to represent different water types.
2.  **Add a "Domain Loss"**: In `src/modules/losses/combined.py`, you could add a new grading function that checks if the model is confusing different water types.
3.  **Tweak the Data Loader**: In `src/data/image_enhancement.py`, you might need to load "unpaired" data (where you don't have a perfect ground truth) if you are working with real thermal images.
