# NeuralDMD
NeuralDMD for Imaging from Sparse Observations


--------------------------------------------------------------------------------------------------------------
Installation
--------------------------------------------------------------------------------------------------------------
After cloning this repository, create a virtual environment and install requirements:
```
python -m venv .neuraldmd_env
. .neuraldmd_env/bin/activate
pip install -r requirements.txt
```

For jax, you might need to install the cuda version via the following:
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

--------------------------------------------------------------------------------------------------------------
Getting Started
--------------------------------------------------------------------------------------------------------------
This repository, we have included the following experiments:
1. Sparse Observations in the Pixel Domain
2. Sparse Observations in the Fourier Domain

1. Pixel Domain: We use April 1-7, 2025 and fit a NeuralDMD. The data available as "./tutorial/pixel/data/data_stream-oper_stepType-instant.nc."
- "/tutorial/pixel/train_model.py" fits a NeuralDMD to sparse pixel observations. After training, the model is saved under "/tutorial/Image/pixel/trained_model.eqx".
- "/tutorial/pixel/test_model.py" visualizes the modes and spectrum learned by the model, and saves a gif of the video reconstructed by these modes and spectrum, and saves the reconstruction data.

2. Fourier Domain: As a simple case, here we use an orbital hotspot for this experiment (/tutorial/generate_data.ipynb creates the orbiting hotspot data with the ngEHT coverage).
- "/tutorial/Fourier/train_model.py" fits a NeuralDMD to sparse Fourier observations. After training, the model is saved under "/tutorial/Fourier/models/trained_model.eqx".
- "/tutorial/Fourier/test_model.ipynb" visualizes the modes and spectrum learned by the model, and saves a gif of the video represented by these modes and spectrum.

All the code for NeuralDMD, Fourier or pixle domain, are contained within the "./tutorial" directory. 

@ Ali SaraerToosi, University of Toronto, 2025