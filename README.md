# NeuralDMD

NeuralDMD fuses classic dynamic mode decomposition (DMD) with neural implicit fields to reconstruct full‑resolution spatiotemporal data from sparse pixel samples or incomplete Fourier (visibility) measurements.

Key features
----------------------
- Reconstruct images, videos, or volumes from highly undersampled measurements (< 1 % pixels or sparse visibilities)

- Provide interpretable spatial modes and temporal spectrum.

- Train on CPU or GPU through JAX (CUDA 11.8+ supported)

Requirements
----------------------
See requirements.txt for the full list.

Installation

```# clone
git clone git@github.com:as2c/NeuralDMD.git
cd NeuralDMD

# (optional) virtual environment
python -m venv .neuraldmd_env
source .neuraldmd_env/bin/activate              # Windows: .venv\Scripts\activate

# core dependencies
pip install -r requirements.txt

# GPU acceleration (replace cuda12_pip with cuda11_pip if needed)
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Repository layout

```
neuraldmd/
 ├─ neural_dmd/           # core library
 ├─ tutorial/
 │   ├─ pixel/            # sparse‑pixel experiment (Apr 1–7 2025 weather data)
 │   └─ fourier/          # sparse‑visibility experiment (orbiting hotspot)
 └─ requirements.txt
```

Quick start

Pixel‑domain example
----------------------
```
cd tutorial/pixel
python train_model.py    # train on 10 % random pixels
after training:
python test_model.py     # plot modes/spectrum and save GIF/MP4
```

Fourier‑domain example
----------------------
```
cd tutorial/fourier
python train_model.py    # train on synthetic visibilities
```
Open "test_model.ipynb" in Jupyter to visualise the results.

Both workflows write a checkpoint (*.eqx) and an outputs/ folder containing plots, videos, and NumPy arrays.
Custom data workflow
Convert your sequence (images or visibilities) to NumPy .npy or NetCDF.
Place it under tutorial/<new_expt>/data/.
Adjust parameters in train_model.py (rank, learning rate, mask).
Run the training and testing scripts as above.