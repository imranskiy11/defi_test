DEFI TEST

Description ~~~~

.
├── config/
│   ├── train.yaml
│   ├── processing.yaml
│   └── inference.yaml
├── data/
├── notebooks/
│   └── eda.ipynb
└── src/
    ├── preprocessing.py
    ├── dataset.py
    ├── features.py
    └── baseline.py

1. config/

Contains YAML configuration files (train.yaml, processing.yaml, inference.yaml) which will eventually define parameters for training, data processing, and inference.

2. data/

Folder for storing raw and processed data files (e.g., pickled pandas DataFrames, CSVs, etc.). In this project, it includes san_tokens_PA_data_100.pkl for demonstration.

3. notebooks/
    eda.ipynb: Jupyter notebook illustrating exploratory data analysis (EDA) of the dataset, including descriptive statistics and basic visualizations.

4. src/

Contains the core source code:

    preprocessing.py: Functions for data cleaning, handling missing values, and other preprocessing steps.
    dataset.py: Functions for loading and preparing datasets, such as build_features_for_all_tokens().
    features.py: Feature engineering logic, including creation of lag features, rolling statistics, etc.
    baseline.py: Main script demonstrating:
        Data loading
        Time-series splitting into train/validation/test
        LightGBM model training (including grid search and time-series cross-validation)
        Evaluation with metrics (RMSE, MAE, R², MAPE, SMAPE)

How to Run

    Install required packages (if not already installed):

pip install -r requirements.txt



Ensure data availability in the data/ folder. For example, san_tokens_PA_data_100.pkl should be placed there.

Run the baseline script (from the project root):

python src/baseline.py

This will:

    Load and preprocess the dataset
    Generate features
    Perform time-series cross-validation
    Perform a grid search for LightGBM hyperparameters
    Evaluate on the test set, printing final metrics in the console