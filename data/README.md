# Dataset setup

This project uses data from the [Paddy Doctor: Paddy Disease Classification competition](https://www.kaggle.com/competitions/paddy-disease-classification). Before downloading or using it, review the [competition data page](https://www.kaggle.com/competitions/paddy-disease-classification/data) and [official rules](https://www.kaggle.com/competitions/paddy-disease-classification/rules).

Kaggle lists 10,407 labeled training images across ten categories and 3,469 test images. The training images are stored in class-named folders. `train.csv` contains `image_id`, `label`, `variety`, and `age`; the competition test images have no labels for local evaluation.

## Expected local layout

Download and extract the competition files so the labeled image folders are inside `train_images/`:

~~~text
data/
├── README.md
└── raw/
    └── paddy-disease-classification/
        ├── train.csv
        ├── sample_submission.csv
        ├── train_images/
        │   └── <class-name>/
        │       └── <image-id>.jpg
        └── test_images/
            └── <image-id>.jpg
~~~

The training loader accepts either the competition data root or its `train_images/` directory. It reads class folders with `torchvision.datasets.ImageFolder` and saves the class-to-index mapping to `results/class_to_idx.json` after training.

On Kaggle, the competition root is `/kaggle/input/competitions/paddy-disease-classification`. Pass it to the training script with `--data-root`.

Keep the downloaded images on your machine. `.gitignore` excludes `data/raw/` and model checkpoint files.
