# Thermal-Gaussian-OMMG

## Description

In this program, we provide One Multi-Modal Gaussian(OMMG) method. We modified the diff-gaussian-rasterization and added a new spherical harmonic function (SH) to represent thermal infrared information, so that the final generated model can render both color and thermal scenes. smoothloss and multimodal regularization were also added to improve rebuild quality and reduce storage costs.

## Setup
The environment required for OMMG is the same as in the main branch, except for diff-gaussian-rasterization,if you have successfully configured the environment according to the main branch, you do not need to recreate the environment, just reinstall diff-gaussian-rasterization.

Since we modified the diff-gaussian-rasterization, you need to reinstall the subdependency, As follow:

```
conda activate thermal_gaussian     #If you have successfully configured the environment with the main branch
pip install submodules/diff-gaussian-rasterization.
```

## Training and Evaluation

```
python train_OMMG.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model>
python render.py -m <path to trained model>  # Generate renderings
python metrics.py -m <path to trained model>  # Compute error metrics on renderings
```
