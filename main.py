from app.loader import load_models
from app.utils import build_applicant_from_dict
from app.predict import two_stage_prediction
import yaml
config = yaml.safe_load(open('config.yaml'))
#Load models
cls_path = "C:\\Users\\rohal\\Downloads\\Loan Approval System\\models\\stage_1_rf_classifier_pipeline.pkl"
reg_path = "C:\\Users\\rohal\\Downloads\\Loan Approval System\\models\\stage_2_rf_regressor_pipeline.pkl"
cls,reg = load_models(config)
def run_cli():
     data = {
        'no_of_dependents': int(input("Enter number of dependents: ")),
        'education': 'Graduate',
        'self_employed': 'No',
        'income_annum':100000,
        'loan_amount': 10000000,
        'loan_term': 12,
        'cibil_score': 600,
        'residential_assets_value': 5000000,
        'commercial_assets_value': 0,
        'luxury_assets_value': 0,
        'bank_asset_value': 0,
    }
     df = build_applicant_from_dict(data, list(cls.feature_names_in_))
     print(two_stage_prediction(cls,reg,df))
if __name__ == "__main__":
    run_cli()