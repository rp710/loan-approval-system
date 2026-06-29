# Loan Approval System


## Overview
The Loan Approval Prediction System is a two-stage machine learning application designed to evaluate loan applicants. 
1. **Stage 1 (Classification):** Predicts whether a loan application should be approved or rejected based on the applicant's profile[cite: 1, 2].
2. **Stage 2 (Regression):** If the classification stage approves the applicant, the system predicts the appropriate loan amount to grant[cite: 1, 2]. 

## Project Structure
* **`app/loader.py`**: Contains the `load_models` function to read the pre-trained `.pkl` model files from the disk using `joblib`.
* **`app/predict.py`**: Houses the `two_stage_prediction` logic that processes an applicant's data through the classification model and conditionally through the regression model.
* **`app/utils.py`**: Contains `build_applicant_from_dict`, a helper function to convert dictionary inputs into a formatted pandas DataFrame and check for missing columns.
* **`analyzer.ipynb`**: A Jupyter Notebook used for exploratory data analysis, preprocessing, and training the machine learning models[cite: 2].
* **`main.py`**: The entry point script that takes user input and prints the prediction results.
* **`models/`**: Directory where the trained pipeline `.pkl` files are saved[cite: 2].
* **`achive/`**: Directory containing the original `loan_approval_dataset.csv` data[cite: 2].

## Data Features
The models are trained on the following applicant features[cite: 2]:
* `no_of_dependents`[cite: 2]
* `education` (Graduate / Not Graduate)[cite: 2]
* `self_employed` (Yes / No)[cite: 2]
* `income_annum`[cite: 2]
* `loan_term`[cite: 2]
* `cibil_score`[cite: 2]
* `residential_assets_value`[cite: 2]
* `commercial_assets_value`[cite: 2]
* `luxury_assets_value`[cite: 2]
* `bank_asset_value`[cite: 2]

## Machine Learning Models
The machine learning pipeline implements `scikit-learn`'s `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` for data preprocessing[cite: 2]. 
* **Classifier**: A `RandomForestClassifier` trained to predict the binary `loan_status` (Approved/Rejected)[cite: 2]. It is saved as `stage_1_rf_classifier_pipeline.pkl`[cite: 2].
* **Regressor**: A `RandomForestRegressor` trained exclusively on approved applicants to predict `loan_amount`[cite: 2]. It is saved as `stage_2_rf_regressor_pipeline.pkl`[cite: 2].

## Setup & Installation
1. Create venv
-uv venv
-uv pip install -r requirements.txt
2. Put your trained `stage_1...pkl` and `stage_2...pkl` files in `models\`.
3. Run locally:
- CLI : `uv run python main.py`

## Config 
See `config.yaml` for runtime parameters(models paths)

## Note:
- make sure the version used to create the model is same as your local environment where you are testing the main.py or streamlit app.
