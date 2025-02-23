

Since we modified the FF-Gaussian-Rasterization, you need to reinstall the subdependency, which can be installed using "pip install submodules/diff-gaussian-rasterization".

In this program, we provide One Multi-Modal Gaussian(OMMG) method. We modified the diff-gaussian-rasterization and added a new spherical harmonic function (SH) to represent thermal infrared information, so that the final generated model can render both color and thermal scenes. smoothloss and multimodal regularization were also added to improve rebuild quality and reduce storage costs.
