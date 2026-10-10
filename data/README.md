# Dataset Setup

The project uses the [Paddy Doctor: Paddy Disease Classification competition](https://www.kaggle.com/competitions/paddy-disease-classification) identified by the author. Review the [competition data page](https://www.kaggle.com/competitions/paddy-disease-classification/data) and [official rules](https://www.kaggle.com/competitions/paddy-disease-classification/rules) before downloading or using the data.

Kaggle documents 10,407 labeled training images across ten categories, plus 3,469 competition test images. The training images are organized in class-named folders. train.csv includes image_id, label, variety, and age; the competition test images do not have labels for local evaluation.

## Expected local layout

Download and extract the competition files so the labeled folders are under train_images/:

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

The training loader accepts the competition data root or its train_images/ subdirectory. It discovers folder names and class indices with torchvision.datasets.ImageFolder; the resulting mapping is written to results/class_to_idx.json after training.

In the supplied Kaggle environment, the competition root is /kaggle/input/competitions/paddy-disease-classification. Pass this path with the --data-root option when running the training script there.

Keep downloaded image data local. The repository's .gitignore excludes data/raw/ and model checkpoint files.
