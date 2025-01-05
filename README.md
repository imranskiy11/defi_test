# DEFI TEST

**Description ~~~~**  
Short description of the project focusing on time-series data preparation, feature engineering, and LightGBM model training for crypto asset price analysis.

## Project Structure
├── config/
│   ├── train.yaml 
│   ├── processing.yaml 
│   └── inference.yaml 
├── data/ 
├── notebooks/ 
│   │ 
│   └── eda.ipynb
└── src/ 
    ├── preprocessing.py 
    ├── dataset.py 
    ├── features.py 
    └── baseline.py
    
    
`config/`  
Contains YAML configuration files (`train.yaml`, `processing.yaml`, `inference.yaml`) which will eventually define parameters for training, data processing, and inference. Currently, they are placeholders.

`data/`  
Folder for storing raw and processed data files (e.g., pickled pandas DataFrames, CSVs, etc.). In this project, it includes `san_tokens_PA_data_100.pkl` for demonstration.

`notebooks/`  
- `eda.ipynb`: Jupyter notebook illustrating exploratory data analysis (EDA) of the dataset, including descriptive statistics and basic visualizations.

`src/`  
Contains the core source code:
- `preprocessing.py`: Functions for data cleaning, handling missing values, and other preprocessing steps.
- `dataset.py`: Functions for loading and preparing datasets, such as `build_features_for_all_tokens()`.
- `features.py`: Feature engineering logic, including creation of lag features, rolling statistics, etc.
- `baseline.py`: Main script demonstrating:
  1. Data loading  
  2. Time-series splitting into train/validation/test  
  3. LightGBM model training (including grid search and time-series cross-validation)  
  4. Evaluation with metrics (RMSE, MAE, R², MAPE, SMAPE)

## How to Run
1. **Install required packages** (if not already installed):
   ```bash
   pip install -r requirements.txt
   
   
Adjust if you have a custom environment setup; the key packages are pandas, numpy, lightgbm, scikit-learn, etc.)

    Ensure data availability in the data/ folder. For example, san_tokens_PA_data_100.pkl should be placed there.

    Run the baseline script (from the project root):
    
    
python src/baseline.py


This will:

    Load and preprocess the dataset
    Generate features
    Perform time-series cross-validation
    Perform a grid search for LightGBM hyperparameters
    Evaluate on the test set, printing final metrics in the console