# Regularity of Images Generated by Convolutional Neural Networks
In this repository we provide the source code to reproduce the U-net results from the paper [A Note on the Regularity of Images Generated by Convolutional Neural Networks](https://arxiv.org/pdf/2204.10588.pdf) by Andreas Habring and Martin Holler . We only provide the code to reproduce the U-net results, as the deep image prior results were obtained by employing the already [published code](https://dmitryulyanov.github.io/deep_image_prior) provided by the authors of the paper.

## Requirements
* Python 3 (versions >=3.9, <3.12).
* [Poetry](https://python-poetry.org/docs/) is used for dependency management.
* `wget` is used to download the pre-trained U-nets automatically.

## Reproduction of the results
 Clone the repository and in the root folder of the repository run the following command from your shell:
```
poetry install
```
This should install all necessary dependencies. Afterwards, from the root directory of the repository run the command
```
poetry run python evaluate_unet.py
```
This will reproduce the U-net results and store them in the folder `data/results/unet`. When running the code for the first time, this might take a while as the pre-trained U-nets will be downloaded from [Zenodo](https://zenodo.org/record/7784039#.ZCVGjR5ByXI). Among the stored result files you will find individual images of all reconstructions as well as two images containing a comparison of all results similar to the ones shown in the paper. The comparison images are named `comparison_resolutions.png` and `comparison_weight_decays.png`.

## Network Training

While we provide the pre-trained U-nets via [Zenodo](https://zenodo.org/record/7784039#.ZCVGjR5ByXI), it is also possible to train the nets yourself. Simply run 
```
poetry run python unet_training.py
```
from the root directory of the repository. Note, however, that this requires a decent GPU and will take some time (more than a day). You can change the training hyperparameters within the file, e.g., train with fewer epochs, but this might lead to different results.

## U-net
The code for the U-net contained in the folder `models/trunks` was mostly taken from [Git repository](https://github.com/aangelopoulos/im2im-uq) which contains the code to reproduce the results of the paper [Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging](https://arxiv.org/abs/2202.05265) by Angelopoulos, Anastasios et al.

## Authors

* **Andreas Habring** andreas.habring@uni-graz.at
* **Martin Holler** martin.holler@uni-graz.at 

## Acknowledgements

Martin Holler is a member of [NAWI Graz](https://www.nawigraz.at) and [BioTechMed Graz](https://biotechmedgraz.at).

## Citation

```
@article{habring2022note,
  title={A Note on the Regularity of Images Generated by Convolutional Neural Networks},
  author={Habring, Andreas and Holler, Martin},
  year={2022},
  journal={SIAM Journal on Mathematics of Data Science}
}
```

## License

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
