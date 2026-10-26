# Rice Leaf Disease Classification with DenseNet

A PyTorch image-classification project based on a custom DenseNet-style model for rice leaf disease images from the [Paddy Doctor Kaggle competition](https://www.kaggle.com/competitions/paddy-disease-classification).

> **Implementation status:** The original notebook had incomplete or misspelled model definitions. The `src/` modules fix those issues and keep the custom DenseNet architecture. The updated pipeline augments training images only and uses a stratified validation split, AdamW, and a learning-rate scheduler. It has not been run, so its effect on accuracy and loss is unknown. The one-epoch result below came from the earlier pipeline and is not directly comparable. The Kaggle outputs and dataset are not included in this repository.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [DenseNet Architecture](#densenet-architecture)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Training](#training)
- [Evaluation Metrics and Results](#evaluation-metrics-and-results)
- [Inference](#inference)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

This project uses a custom DenseNet-style model in PyTorch to classify rice leaf images by disease. It also explores whether image classification could help farmers in Central Vietnam notice disease symptoms earlier.

The model is intended to support decision-making. It has not been evaluated on field images and should not replace an agricultural expert.

## Problem Statement

Rice leaf symptoms can be hard to distinguish by eye. This project tests whether a deep learning model can classify images using the labels in the Paddy Doctor dataset. The competition provides labeled training images and a separate test set without labels.

## Dataset

The dataset comes from [Paddy Doctor: Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification). Kaggle lists 10,407 labeled training images across ten categories and 3,469 competition test images. The training images are stored in class-named folders, and `train.csv` contains `image_id`, `label`, `variety`, and `age`. Check the [competition data page](https://www.kaggle.com/competitions/paddy-disease-classification/data) and [official rules](https://www.kaggle.com/competitions/paddy-disease-classification/rules) for access and usage terms.

The loader expects labeled images under `train_images/` and reads them with `torchvision.datasets.ImageFolder`. It saves the class-to-index mapping to `results/class_to_idx.json` after training. The competition test images have no labels, so the project does not use them for local evaluation.

The training pipeline sets aside `int(0.2 * N)` images for validation and uses the rest for training. The split is stratified by class and uses seed 123. The dataset is not included in this repository; see [data/README.md](data/README.md).

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib (used in the original notebook; the refactored scripts do not generate plots)

The notebook does not record package versions or a Python version, so `requirements.txt` lists the packages without version pins. The refactored modules use type-hint syntax from Python 3.10 or newer.

## DenseNet Architecture

The notebook defines a custom `DenseNetB121`; it does not use torchvision's pretrained DenseNet. Each dense block adds newly generated feature maps to those already in the block. A bottleneck first uses a 1x1 convolution to expand to `4 * growth_rate` channels, then a 3x3 convolution to produce `growth_rate` new channels. Between dense blocks, transition layers apply BatchNorm and ReLU, halve the channel count with a 1x1 convolution, and downsample with average pooling.

| Component | Configuration read from the notebook |
| --- | --- |
| Input | RGB image resized to 224 × 224 |
| Stem | BatchNorm → ReLU → 7x7 convolution (64 channels, stride 2) → 3x3 max pooling (stride 2) |
| Dense blocks | 6, 12, 24, and 16 bottleneck layers |
| Growth rate | 32 channels |
| Transitions | After the first three blocks; channel count and spatial dimensions are each halved |
| Head | Adaptive global average pooling → fully connected layer → raw logits |
| Classes | 10, matching the competition training folders |

The model keeps the notebook's initial BatchNorm/ReLU/convolution order. The code also fixes its missing `DenseBlock` forward method, class-name mismatches, misspelled third-transition references, and lowercase `nn.linear` typo. The notebook contains no usable checkpoint, so compatibility with previously trained weights is unverified.

## Project Structure

~~~text
.
├── data/
│   └── README.md
├── notebooks/
│   ├── rice_densenet_training.ipynb
│   └── original/
│       └── qmanh.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
~~~

The repository root also contains `machine-learning-workflow.html`, a standalone diagram of a typical ML pipeline.

Training creates `results/densenetb121_paddy.pth`, `results/history.json`, and `results/class_to_idx.json`. The original notebook remains at `notebooks/original/qmanh.ipynb`. The runnable notebook, `notebooks/rice_densenet_training.ipynb`, can predict one image and optionally create a Kaggle test-set submission. There is no separate `predict.py` module.

## Installation and Setup

From the repository root, create a virtual environment and install the listed packages:

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
~~~

Use Python 3.10 or newer. Place the competition files in the layout described in [data/README.md](data/README.md). The default loader path is data/raw/paddy-disease-classification/; pass either the competition root or its train_images/ directory. A CUDA GPU is used when available; otherwise, the script selects CPU. Compatible PyTorch/Torchvision versions still need to be selected for the target environment.

## Training

Training runs for 10 epochs by default, with batch size 32 and four data-loader workers. It uses AdamW (learning rate 0.001, `eps=1e-8`, weight decay `1e-4`), cross-entropy loss, and `ReduceLROnPlateau`, which lowers the learning rate when validation loss stops improving. Only training images receive horizontal flips, small rotations, and mild color jitter. All images are resized to 224 x 224 and normalized with ImageNet channel statistics. The validation split is stratified and seeded. The code saves the checkpoint with the highest validation accuracy; if epochs tie, it keeps the one with lower validation loss. The updated pipeline has not been run, so the effect of these settings on generalization is unknown.

Run from the repository root:

~~~bash
python -m src.train --data-root data/raw/paddy-disease-classification --epochs 10
~~~

On Kaggle, pass the competition path you provided:

~~~bash
python -m src.train --data-root /kaggle/input/competitions/paddy-disease-classification --epochs 10
~~~

The code saves the model state dictionary with the best validation accuracy, the epoch history (including the selected epoch), and the class mapping under `results/`. The modular command-line entry point has not been run independently.

## Evaluation Metrics and Results

After each epoch, training records cross-entropy loss and accuracy for both subsets. `src.evaluate` uses the same seeded validation split and reports validation loss and accuracy.

An earlier Kaggle run of `notebooks/rice_densenet_training.ipynb` used CUDA and completed one epoch on the 10,407 labeled images. It used a random split with 8,326 training images and 2,081 validation images. The custom model had 6,912,818 parameters. This run predates the augmentation, stratified split, and optimizer settings now in the code.

| Metric | Training | Validation |
| --- | ---: | ---: |
| Loss | 1.6619 | 1.3716 |
| Accuracy | 42.37% | 56.37% |

That epoch took 81.5 seconds. These are holdout metrics from the earlier one-epoch run, not a Kaggle leaderboard score or results from the updated 10-epoch pipeline. The notebook also ran inference on the competition test images and created `submission.csv`, but the file was not submitted. The checkpoint, history, class mapping, and submission remain in the Kaggle session at `/kaggle/working/results/`; they are not part of this repository. The updated code has not produced new results.

After training, evaluate the saved model with:

~~~bash
python -m src.evaluate \
  --data-root data/raw/paddy-disease-classification \
  --checkpoint results/densenetb121_paddy.pth
~~~

## Inference

The notebook at `notebooks/rice_densenet_training.ipynb` provides `predict_image()` for a single image. It can also create `submission.csv` when `test_images/` and `sample_submission.csv` are available. The Kaggle run exercised test-set inference, but its submission file has not been added here or submitted to Kaggle.

## Limitations

- The updated training pipeline has not been run; its effect on validation metrics is unmeasured.
- The custom DenseNet is trained from scratch and may need more than 10 epochs for convergence.
- The competition dataset and generated checkpoint/results are not included in this repository; they remain in the Kaggle session.
- Performance on field images, including images from Central Vietnam, has not been evaluated.
- Similar disease symptoms, lighting, background, camera angle, and image quality may affect predictions.
- Predictions should support awareness and follow-up, not replace assessment by an agricultural expert.

## Future Improvements

- Run the updated 10-epoch pipeline and record its results before comparing it with the historical baseline.
- Add per-class precision, recall, F1-score, and a confusion matrix to the evaluation report.
- Evaluate on representative field images collected in Vietnam.
- Add a dedicated `predict.py` command and verify checkpoint reload and class mapping.
- Pin compatible Python, PyTorch, and Torchvision versions once the target environment is chosen.

## Author

- **Name:** Nguyễn Quốc Mạnh
- **GitHub:** [nguyenquocmanh12012008-tech](https://github.com/nguyenquocmanh12012008-tech)
- **Kaggle notebook:** [DenseNet training notebook](https://www.kaggle.com/code/qucmnhh/notebook11b63cb064/edit)
