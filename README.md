# THERMALGAUSSIAN: THERMAL 3D GAUSSIAN SPLATTING

Rongfeng Lu<sup>1,3,∗,†</sup> , Hangyu Chen<sup>1,∗</sup> , Zunjie Zhu<sup>1,3</sup> , Yuhang Qin<sup>1</sup> , Ming Lu<sup>2</sup> , Le Zhang<sup>1</sup> , Chenggang Yan<sup>1</sup> , Anke Xue<sup>1,†</sup>

<sup>1</sup>Hangzhou Dianzi University, <sup>2</sup>Intel Labs China, <sup>3</sup>Lishui Institute of Hangzhou Dianzi University

### [[Paper](https://arxiv.org/abs/2409.07200)] | [[Dataset](https://drive.google.com/drive/folders/1xuPwqezNW_LSB5C-H8TQFJ7nLoSS2BOg?usp=drive_link)] | [[Project Page](https://thermalgaussian.github.io/)]

## Abstract:
*Thermography is especially valuable for the military and other users of surveillance cameras. Some recent methods based on Neural Radiance Fields (NeRF) are proposed to reconstruct the thermal scenes in 3D from a set of thermal and RGB images. However, unlike NeRF, 3D Gaussian splatting (3DGS) prevails due to its rapid training and real-time rendering. In this work, we propose ThermalGaussian, the first thermal 3DGS approach capable of rendering high-quality images in RGB and thermal modalities. We first calibrate the RGB camera and the thermal camera to ensure that both modalities are accurately aligned. Subsequently, we use the registered images to learn the multimodal 3D Gaussians. To prevent the overfitting of any single modality, we introduce several multimodal regularization constraints. We also develop smoothing constraints tailored to the physical characteristics of the thermal modality. Besides, we contribute a real-world dataset named RGBT-Scenes, captured by a hand-hold thermal-infrared camera, facilitating future research on thermal scene reconstruction. We conduct comprehensive experiments to show that ThermalGaussian achieves photorealistic rendering of thermal images and improves the rendering quality of RGB images. With the proposed multimodal regularization constraints, we also reduced the model’s storage cost by 90%. Our project page is at https://thermalgaussian.github.io/.*


## Overview

Our work is based on improvements in [3D gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), so software and hardware configurations and other details can be referred to its work.


## Setup

### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate thermal_gaussian
```
## Data Preparation
You could download our own dataset RGBT-Scenes directly from [Google Drive](https://drive.google.com/drive/folders/1xuPwqezNW_LSB5C-H8TQFJ7nLoSS2BOg?usp=drive_link).

Because our work can realize the reconstruction of color and temperature multimodal scenes，so We read the color image and the thermal image at the same time, and you need to manually divide the train set and the test set, that is, there are two folders named "rgb" and "thermal" to store the color image and the thermal image respectively, and each folder is divided into "test" and "train" folders. And we put the result after colmap in colmap/sparse.

If you want to run our program, make sure to put the "sparse" folder in the same directory as the "rgb" folder and the "thermal" folder, as follows:

```
<location>
|---rgb
|   |---test
|   |   |---<image 0>
|   |   |---<image 8>
|   |   |---...
|   |---train
|       |---<image 1>
|       |---<image 2>
|       |---...
|---thermal
|   |---test
|   |   |---<image 0>
|   |   |---<image 8>
|   |   |---...
|   |---train
|       |---<image 1>
|       |---<image 2>
|       |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

## Running

To train and evaluate the model, use the following command-line arguments:

- `-s` to specify the input path (required)
- `-m` to specify the output path (optional)

The input path must be provided, while the output path is optional. If you do not specify an output directory (`-m`), trained models will be saved in folders with randomized unique names inside the `output` directory.

## Training and Evaluation

Run the following commands in sequence:

For MSMG version:

```
python train_MSMG.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model>
python render.py -m <path to trained model>  # Generate renderings
python metrics.py -m <path to trained model>  # Compute error metrics on renderings
```

For MFTG version:

```
python train_MFTG.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model>
python render.py -m <path to trained model>  # Generate renderings
python metrics.py -m <path to trained model>  # Compute error metrics on renderings
```

For OMMG version:

Switch to the OMMG branch of this project because the code is different from the MSMG and MFTG versions, and reinstall submodules/diff-gaussian-rasterization.

- `train_MSMG.py` → **Multiple Single-Modal Gaussians (MSMG)**
- `train_MFTG.py` → **Multimodal Fine-Tuning Gaussians (MFTG)**
- `train_OMMG.py` → **One Multi-Modal Gaussian (OMMG)**

## Example

For MSMG version:

```
python train_MSMG.py -s data/Truck -m output/Truck
python render.py -m output/Truck
python metrics.py -m output/Truck
```

For MFTG version:

```
python train_MFTG.py -s data/Truck -m output/Truck
python render.py -m output/Truck
python metrics.py -m output/Truck
```

These examples train models on the `data/Truck` dataset, save the models in `output/Truck`, generate renderings, and compute error metrics based on the renderings.

Note that similar to MipNeRF360, we target images at resolutions in the 1-1.6K pixel range. For convenience, arbitrary-size inputs can be passed and will be automatically resized if their width exceeds 1600 pixels. We recommend to keep this behavior, but you may force training to use your higher-resolution images by setting ```-r 1```.


## Processing your own Scenes

Our COLMAP loaders expect the following dataset structure in the source path location:
```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

If you want to make your own scene data, you need to put your captured images into the input folder, as follows
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
Then run:
```shell
python convert.py -s <location> 

#such as : 
#python convert.py -s data/Truck
```

## Citation

If you use this project in your research, please cite it as follows:

```
@article{lu2024thermalgaussian,
  title={ThermalGaussian: Thermal 3D Gaussian Splatting},
  author={Lu, Rongfeng and Chen, Hangyu and Zhu, Zunjie and Qin, Yuhang and Lu, Ming and Zhang, Le and Yan, Chenggang and Xue, Anke},
  journal={arXiv preprint arXiv:2409.07200},
  year={2024}
}
```

