# Summary 

Points of Interest (POIs) are published in an application where users can engage with them through voting (Like/Dislike) or bookmarking. These interactions serve as indicators of POI engagement.

This work leverages available POI data to develop classification models that predict whether a POI will achieve high or low engagement. Each POI includes both tabular metadata and an associated image.

After evaluating several classical machine learning and deep learning models using a protocol based on Optuna library, two models were selected for production use in identifying high-engagement POIs.

A Random Forest model was trained and demonstrates the ability to identify all high-engagement POIs using tabular metadata, with the trade-off of including some false positives.

A second model featuring a multimodal architecture was developed, combining tabular metadata with POI images. This model achieves a balanced performance between sensitivity (0.88) and precision (0.84).

Detailed results are available in the `notebooks` directory: see **exerciseDL.pdf** or **exerciseDL.html**

# Methodology

Hyperparameter optimization was performed using the **run_optuna_ML.py** and **run_optuna_DL.py** scripts, which execute the optimization procedures defined in the configuration files stored in the `optuna/run_config` directory.

The methodology was designed to streamline the optimization process by enabling easy configuration of models, search spaces, and scoring metrics. All optimization results are saved in the `optuna/file_config` directory for use in subsequent steps. These optimized parameters are then utilized in the model training phase (**training_model_ML.py** and **training_model_DL.py**) to build production-ready models. The trained models can subsequently be deployed for inference using the **model_prediction_ML.py** and **model_prediction_DL.py** scripts. This comprehensive protocol ensures full traceability across all studies and experimental runs.

# Code Structure

The scripts for model optimization and the reusable modules containing the functions and classes used in the tuning process are shown here.

```
.
├── data
├── environment.yaml
├── models
├── notebooks
├── optuna
│   ├── file_config
│   ├── run_config
|   └── optuna_DL_exercise.db
├── README.md
└── src
    ├── model_prediction_DL.py
    ├── model_prediction_ML.py
    ├── run_optuna_DL.py
    ├── run_optuna_ML.py
    ├── training_model_DL.py
    ├── training_model_ML.py
    ├── utilsDataset.py
    ├── utilsFT.py
    ├── utilsModel.py
    ├── utilsNN.py
    ├── utilsOptuna.py
    ├── utilsPlots.py
    ├── utilsPreproc.py
    └── utils.py

```

# Reproducibility

Create an environment with conda (version 25.1.1)

```
conda env create -f environtment.yaml
conda activate exerciseDL
```

# Create HTML and PDF files

Quarto version 1.8.24

```
quarto render exerciseDL.ipynb --to html
quarto render exerciseDL.ipynb --to pdf
```
