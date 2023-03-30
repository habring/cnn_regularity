# Regularity of Images Generated by Convolutional Neural Networks
In this repository we provide the source code to reproduce the U-net results from the paper [A Note on the Regularity of Images Generated by Convolutional Neural Networks](https://arxiv.org/pdf/2204.10588.pdf) by Andreas Habring and Martin Holler . We only provide the code to reproduce the U-net results, as the deep image prior results were obtained by employing the already [published code](https://dmitryulyanov.github.io/deep_image_prior) provided by the authors of the paper.

## Requirements
The code is written for Python 3.9 (versions >=3.9, <3.12). Dependency management is handled with [poetry](https://python-poetry.org/docs/). For details on necessary package versions see the file `pyproject.toml`. Before using the code make sure you have installed poetry on your system.

## Reproduction of the results
 Clone the repository and in the root folder of the repository run the following command from your shell:
```
poetry install
```
This should install all necessary dependencies. Afterwards, from the root directory of the repository run the command
```
poetry run python evaluate_unet.py
```
This will reproduce the u-net results and store them in the folder `data/results/unet`. When running the code for the first time, this might take a while as the pre-trained u-nets will be downloaded from [Zenodo](https://zenodo.org/record/7784039#.ZCVGjR5ByXI).

## Network Training

While we provide the pre-trained U-nets via [Zenodo](https://zenodo.org/record/7784039#.ZCVGjR5ByXI), it is also possible to train the nets yourself. Simply run 
```
poetry run python evaluate_unet.py
```
from the root directory of the repository. Note, however, that this requires a decent GPU and will take some time (more than a day). You can change the training hyperparameters within the file, e.g., train with fewer epochs, but this might lead to different results.

## Authors

* **Andreas Habring** andreas.habring@uni-graz.at
* **Martin Holler** martin.holler@uni-graz.at 

## Acknowledgements

Martin Holler further is a member of NAWI Graz (https://www.nawigraz.at) and BioTechMed Graz (https://biotechmedgraz.at).

## Citation

```
@article{habring2022note,
  title={A Note on the Regularity of Images Generated by Convolutional Neural Networks},
  author={Habring, Andreas and Holler, Martin},
  journal={arXiv preprint arXiv:2204.10588},
  year={2022},
  journal={SIAM Journal on Mathematics of Data Science}
}
```

## License

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
