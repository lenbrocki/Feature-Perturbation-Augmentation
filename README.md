# Feature-Perturbation-Augmentation

This repository contains the code to reproduce the results of our paper Feature Perturbation Augmentation (FPA): arXiv.  
Requirements for running the notebooks are [PyTorch](pytorch.org), [PyTorch Lightning](www.pytorchlightning.ai) and [Numba](https://numba.pydata.org).

We provide code for training the models with FPA, but pre-trained weights can also be downloaded [here](https://drive.google.com/file/d/1FhAKvLf-2u5LFWxBARy699R8PLOhAJCj/view?usp=share_link) and have to be put in a folder called "weights".

As a first step, importance estimates are generated using the [get_estimators.ipynb](https://github.com/lenbrocki/Feature-Perturbation-Augmentation/blob/main/get_estimators.ipynb) notebook and the perturbation-based evaluation can then be performed using the [perturb.ipynb](https://github.com/lenbrocki/Feature-Perturbation-Augmentation/blob/main/perturb.ipynb) notebook.




