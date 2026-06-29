# import streamlit as st
# from app.loader import load_models
# from app.utils import build_applicant_from_dict
# from app.predict import two_stage_prediction
# import yaml

# with open('config.yaml') as f:
#     config = yaml.safe_load(f)

# st.set_page_config(page_title='Loan approval',layout='centered')
# st.title("Loan Approval - 2 stage prediction")

# cls,reg = load_models(config)
# st.sidebar.header('Model Info')

# try:
#     st.sidebar.write("Classifier expects",list(cls.feature_names_in_))
# except Exception:
#     st.sidebar.write("Classifier feature names unavailable")

# st.header('Applicant details')
# default = config['ui']['default_inputs']


# cols = st.columns(2)
# with cols[0]:
#     no_of_dependents = st.number_input('No of dependents',value=int(default['no_of_dependents']))
#     # no_of_dependents = st.number_input('No of dependents', value=safe_val)
#     education = st.selectbox("Education",options=['Graduate','Not Graduate'],index=0 if default['education']== 'Graduate' else 1)
#     self_employed = st.selectbox('Self Employed',options=['Yes','No'],index=0 if default['self_employed'] == 'Yes' else 1)
#     income_annum = st.number_input('Annual Income',value=float(default['income_annum']))
#     loan_amount = st.number_input('Loan Amount requested',value=float(default['loan_amount']))
# with cols[1]:
#     loan_term = st.number_input('Loan Term',value=float(default['loan_term']))
#     cibil_score = st.number_input('Cibil Score',value=float(default['cibil_score']))
#     residential_assets_value = st.number_input('Residential Asset value',value=float(default['residential_assets_value']))
#     commercial_assets_value  = st.number_input('Commerical Asset Value',value=float(default['commercial_assets_value']))
#     luxury_assets_value = st.number_input('Luxury Asset Value',value=float(default['luxury_assets_value']))
#     bank_asset_value = st.number_input('Bank Asset Value',value=float(default['bank_asset_value']))

# applicant = {
#     'no_of_dependents': no_of_dependents,
#     'education': education,
#     'self_employed': self_employed,
#     'income_annum': income_annum,
#     'loan_amount': loan_amount,
#     'loan_term': loan_term,
#     'cibil_score': cibil_score,
#     'residential_assets_value': residential_assets_value,
#     'commercial_assets_value': commercial_assets_value,
#     'luxury_assets_value': luxury_assets_value,
#     'bank_asset_value': bank_asset_value
# }
# # if st.button('Predict'):
# #     try:
# #         expected_cols = list(cls.feature_names_in_)
# #         applicant_df = build_applicant_from_dict(applicant,expected_cols)
# #         results = two_stage_prediction(cls,reg,applicant_df)
# #         res = results[0] # final dictionary
# #         st.write('Approval Probability : {:2%}'.format(res['approved_proba']))
# #         if res['approved'] == 1:
# #             st.success('Approved')
# #             st.write(f'Predicted loan amount :{res['reg_pred']:.2f}')
# #         else:
# #             st.error('Rejected')
# #     except Exception as e:
# #         st.error(f"Prediction failed: {e}")

# if st.button('Predict'):
#     try:
#         # Build the applicant dataframe
#         applicant_df = build_applicant_from_dict(applicant, list(cls.feature_names_in_))
#         results = two_stage_prediction(cls, reg, applicant_df)
#         res = results[0] 
        
#         st.write(f'Approval Probability : {res["approved_proba"]:.2%}')
        
#         if res['approved'] == 1:
#             st.success('Loan Status: Approved')
#             st.write(f'Approved Loan amount: {res["reg_pred"]:.2f}')
#         else:
#             # Determine if it was a total rejection or just an amount issue
#             requested = float(applicant['loan_amount'])
#             st.error('Loan Status: Rejected')
            
#             if requested > res['reg_pred']:
#                 st.warning(f"Your requested amount ({requested:,.2f}) exceeds your eligible limit.")
            
#             st.info(f'The maximum loan amount you qualify for is: {res["reg_pred"]:,.2f}')
            
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
import streamlit as st
from app.loader import load_models
from app.utils import build_applicant_from_dict
from app.predict import two_stage_prediction
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Added 'layout="wide"' is fine, but removing columns is key for mobile
st.set_page_config(page_title='Loan approval', layout='centered')
st.title("Loan Approval - 2 stage prediction")

cls, reg = load_models(config)
st.sidebar.header('Model Info')

try:
    st.sidebar.write("Classifier expects", list(cls.feature_names_in_))
except Exception:
    st.sidebar.write("Classifier feature names unavailable")

st.header('Applicant details')
default = config['ui']['default_inputs']

# REPLACED: Removed st.columns(2). Inputs will now stack vertically for mobile.
no_of_dependents = st.number_input('No of dependents', value=int(default['no_of_dependents']))
education = st.selectbox("Education", options=['Graduate', 'Not Graduate'], index=0 if default['education'] == 'Graduate' else 1)
self_employed = st.selectbox('Self Employed', options=['Yes', 'No'], index=0 if default['self_employed'] == 'Yes' else 1)
income_annum = st.number_input('Annual Income', value=float(default['income_annum']))
loan_amount = st.number_input('Loan Amount requested', value=float(default['loan_amount']))
loan_term = st.number_input('Loan Term', value=float(default['loan_term']))
cibil_score = st.number_input('Cibil Score', value=float(default['cibil_score']))
residential_assets_value = st.number_input('Residential Asset value', value=float(default['residential_assets_value']))
commercial_assets_value = st.number_input('Commerical Asset Value', value=float(default['commercial_assets_value']))
luxury_assets_value = st.number_input('Luxury Asset Value', value=float(default['luxury_assets_value']))
bank_asset_value = st.number_input('Bank Asset Value', value=float(default['bank_asset_value']))

applicant = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

if st.button('Predict'):
    try:
        # Build the applicant dataframe
        applicant_df = build_applicant_from_dict(applicant, list(cls.feature_names_in_))
        results = two_stage_prediction(cls, reg, applicant_df)
        res = results[0] 
        
        st.write(f'Approval Probability : {res["approved_proba"]:.2%}')
        
        if res['approved'] == 1:
            st.success('Loan Status: Approved')
            st.write(f'Approved Loan amount: {res["reg_pred"]:,.2f}')
        else:
            requested = float(applicant['loan_amount'])
            st.error('Loan Status: Rejected')
            
            if requested > res['reg_pred']:
                st.warning(f"Your requested amount ({requested:,.2f}) exceeds your eligible limit.")
            
            st.info(f'The maximum loan amount you qualify for is: {res["reg_pred"]:,.2f}')
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")