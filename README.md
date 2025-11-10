# PIR Anomaly Detection — Deployment Package

> Tiny footprint anomaly-detection system for PIR / motion-sensor feature vectors.
> Ready-to-deploy TFLite model + scalers + threshold for running on microcontrollers (Raspberry Pi Pico W / MicroPython) or host devices.

---

## Table of contents

* [Project overview](#project-overview)
* [What’s included](#whats-included)
* [Dataset & features](#dataset--features)
* [Model details](#model-details)
* [How it works (high-level)](#how-it-works-high-level)
* [Quick start — Host (Python) inference example](#quick-start---host-python-inference-example)
* [Quick start — Raspberry Pi Pico W (MicroPython) deployment notes](#quick-start---raspberry-pi-pico-w-micropython-deployment-notes)
* [Files & values you should know](#files--values-you-should-know)
* [Evaluation, limitations & suggestions](#evaluation-limitations--suggestions)
* [Development / training notes (if you want to extend)](#development--training-notes-if-you-want-to-extend)
* [License & contact](#license--contact)

---

## Project overview

This repository contains a minimal anomaly-detection package built for PIR (passive infrared) / motion-sensor feature vectors. It uses a very small TFLite model (suitable for constrained hardware) together with precomputed scaler parameters and a threshold for one-class / reconstruction-based anomaly detection. The package is designed to run on microcontrollers like the Raspberry Pi Pico W or on a host (Raspberry Pi / PC) for testing and development.

Key idea: normalize incoming 8-dimensional feature vectors, let the model reconstruct them (autoencoder style), compute reconstruction error, and compare against a threshold to flag anomalies.

---

## What’s included

* `pir_anomaly_model.tflite` — main TFLite model file (tiny; see details). 
* `scaler_mean.npy` — per-feature mean for normalization. 
* `scaler_scale.npy` — per-feature scale (std or scale) for normalization. 
* `threshold.npy` — reconstruction-error threshold used to declare anomalies. 
* `pir_dataset.json` — sample dataset entries / raw readings and computed features (used for training / validation). 

---

## Dataset & features

The dataset contains labelled examples (`"normal"`) with both raw binary PIR readings and derived numeric features. Each datapoint includes an 8-element `features` vector, for example:

```json
"features": [109, 6, 18.17, 29, 70.14, 196, 0.1817, 12]
```

These features are the model input and represent summary statistics computed from raw PIR readings (raw binary sequences are included in each sample). Example records and the full sample JSON are in `pir_dataset.json`. 

---

## Model details

* **Model file:** `pir_anomaly_model.tflite`. 
* **Model size:** **~3.66 KB** (3,752 bytes). 
* **Input shape:** `[1, 8]` — single sample of 8 features. 
* **Output shape:** `[1, 8]` — reconstructed 8 features (autoencoder behavior). 

Because the model maps 8→8, the anomaly score is computed from reconstruction error between input and output.

**Threshold used for anomaly detection:** `0.065759`. 

**Scaler (normalization) parameters** (mean & scale arrays are provided and must be applied before inference): the deployment package includes the `scaler_mean.npy` and `scaler_scale.npy` files; their numeric values are drawn from the deployment metadata. 

---

## How it works (high-level)

1. Collect/preprocess a raw PIR reading window → compute the same 8 summary features used at training time. (See `pir_dataset.json` for examples.) 
2. Normalize features: `(x - mean) / scale` using the provided scaler arrays. 
3. Run the normalized 8-D vector through the TFLite model to get an 8-D reconstruction. 
4. Compute reconstruction error (e.g., mean squared error across 8 dims).
5. If error > `threshold.npy` (0.065759), mark sample as **anomalous**; otherwise **normal**. 

---

## Quick start — Host (Python) inference example

> This example uses `tflite-runtime` or `tensorflow` (if installed). On a constrained device you may use a smaller interpreter or a simplified inference routine (see Pico notes).

```python
# requirements (host): numpy, tflite-runtime (or tensorflow)
# pip install numpy tflite-runtime
import numpy as np
import tflite_runtime.interpreter as tflite  # or use from tensorflow import lite as tflite

# load model + scalers + threshold
interpreter = tflite.Interpreter(model_path="pir_anomaly_model.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")
threshold = np.load("threshold.npy").item()  # scalar

def is_anomaly(features_raw):
    # features_raw: shape (8,) or list of 8 numbers
    x = np.array(features_raw, dtype=np.float32).reshape(1,8)
    x_norm = (x - mean) / scale
    interpreter.set_tensor(input_index, x_norm.astype(np.float32))
    interpreter.invoke()
    recon = interpreter.get_tensor(output_index)   # shape (1,8)
    # reconstruction error (MSE)
    mse = np.mean((x_norm - recon)**2)
    return mse > threshold, float(mse)

# example usage
features = [109, 6, 18.17, 29, 70.14, 196, 0.1817, 12]
anomaly, score = is_anomaly(features)
print("anomaly?", anomaly, "score:", score)
```

**Notes**

* If using full TensorFlow, the code is similar but you import the TF Lite interpreter from `tensorflow.lite`.
* The pipeline assumes you compute exactly the same 8 features as used in training. See `pir_dataset.json` for example calculations. 

---

## Quick start — Raspberry Pi Pico W (MicroPython) deployment notes

The deployment package contains a short guide recommending two approaches: 1) use `micropython-tflite` (experimental) if available, or 2) use a simplified inference approach (hand-coded inference) for production on Pico W. The package explicitly states `TFLite for MicroPython is still experimental` and suggests using the simplified approach in production. 

General steps for Pico W:

1. Copy these files to the Pico W filesystem (via `ampy`, `rshell`, Thonny, or the WebREPL):

   * `pir_anomaly_model.tflite` (if you plan to use a TFLite interpreter), `scaler_mean.npy`, `scaler_scale.npy`, `threshold.npy`. 
2. If using `micropython-tflite`, install it on your Pico (if a build exists for your board). This is experimental — follow that project's instructions. 
3. If not using tflite on-device, use a **simplified inference**:

   * Export the model to a tiny hand-coded function (e.g., compute a few linear layers manually) or embed a lookup/table or very small hand-rolled network. The deployment notes suggest this route for production. 
4. Perform normalization on-device (apply `mean` & `scale`) before inference. 
5. Compute the same reconstruction error and compare with the saved threshold. 

**MicroPython pseudo-code (concept):**

```python
# PSEUDO: not runnable as-is — illustrates the flow
from machine import Pin
import ujson, uos

mean = load_npy_as_list("scaler_mean.npy")  # helper to read or port scaler as a .py array
scale = load_npy_as_list("scaler_scale.npy")
threshold = load_threshold("threshold.npy")

def normalize(x):
    return [(xi - m)/s for xi,m,s in zip(x, mean, scale)]

def simple_infer(x_norm):
    # either call micropython-tflite interpreter or a tiny hand-coded net
    return tflite_invoke(x_norm)  # or manual layers

def is_anomaly(features):
    x_norm = normalize(features)
    recon = simple_infer(x_norm)
    mse = sum((a-b)**2 for a,b in zip(x_norm, recon))/len(x_norm)
    return mse > threshold, mse
```

Because real MicroPython deployment depends on how you port scalers & the model, include a small helper to embed `mean`/`scale` as lists in a `.py` module if reading `.npy` is inconvenient on-device.

---

## Files & values you should know (copied from deployment metadata)

* `pir_anomaly_model.tflite` — **model size:** 3752 bytes. 
* **Input shape:** `[1, 8]`. 
* **Output shape:** `[1, 8]`. 
* **Threshold:** `0.065759`. 
* **Scaler mean example (deployment):** `[175.85, 6.25, 120.997, 126.6, 133.665, 227.4, 0.29309, 12.25]`. 
* **Scaler scale example (deployment):** `[195.6653, 7.8986, 218.3744, 215.7657, 173.4499, 186.4147, 0.3261044, 15.974589]`. 

> Keep these numbers with the model — inference correctness depends on using the exact mean/scale used during training. 

---

## Evaluation, limitations & suggestions

* **What this package provides:** a tiny, deployable detector suitable for constrained hardware (Pico W) using reconstruction-based anomaly scoring. 
* **Limitations:**

  * The model is tiny — tradeoff: reduced capacity may limit sensitivity for complex anomalies. 
  * The deployment metadata does not include explicit training/validation metrics (e.g., ROC/AUC, precision/recall) in the package — if you need those, run evaluation on held-out labelled data.
  * MicroPython TFLite support is experimental — on-device inference may require a hand-ported/simplified model or use of the simplified inference approach suggested in the package. 

**Suggestions to improve:**

* Add a small evaluation notebook showing false-positive / false-negative analysis on a labelled validation set.
* Provide a reference script to compute the 8 input features from raw PIR readings so downstream users reproduce preprocessing exactly (the `pir_dataset.json` contains examples — consider including an explicit `feature_extractor.py`). 
* If deploying to different sensors/environments, re-fit scalers and re-evaluate threshold on new normal data.

---

## Development / training notes (how to extend)

1. Recompute scaler means & scales on new "normal" data and retrain the autoencoder.
2. Tune threshold: compute reconstruction error distribution on validation `normal` set and choose threshold at desired false-positive rate.
3. If more capacity needed, train a slightly larger model and re-export to TFLite with quantization for microcontroller deployment.

---

## License & contact

* Add your preferred license file (e.g., `MIT`, `Apache-2.0`) to the repo.
* For questions / issues, open an issue on this repo or contact the maintainer (add your email / handle here).

---

## Appendix — Example `README` checklist before publishing

* [ ] Include `feature_extractor.py` showing exact feature computations. 
* [ ] Add `requirements.txt` / `environment.yml` for host testing.
* [ ] Add a simple `test_inference.py` that runs the host example on a few samples.
* [ ] Add license and contributor info.

---
