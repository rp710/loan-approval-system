import warnings
# def two_stage_prediction(cls,reg,applicant_df):
#     warnings.filterwarnings("ignore", category=UserWarning)
#     # First stage: Classification
#     proba = cls.predict_proba(applicant_df)
#     preds = cls.predict(applicant_df)
#     results = []
#     approved_idx = 1
#     i=0
#     approved = int(preds[i])
#     approved_prob = float(proba[i,approved_idx])
#     reg_pred = None

#     if approved == 1:
#         # Second stage: Regression
#         applicant_df_reg = applicant_df.copy()
#         applicant_df_reg['loan_status'] = 'Approve'
#         reg_pred = float(reg.predict(applicant_df_reg)[0])
#         results.append({
#             'approved':approved,
#             'approved_proba':approved_prob,
#             'reg_pred':reg_pred
#         })
#     else:
#         print("Loan not approved, skipping regression prediction.")
#     return results

import warnings

def two_stage_prediction(cls, reg, applicant_df):
    """
    Predicts loan approval status and the eligible loan amount.
    The loan is rejected if the classifier says 'No' OR 
    if the requested amount exceeds the regression-predicted eligible amount.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 1. Classification Stage
    proba = cls.predict_proba(applicant_df)
    preds = cls.predict(applicant_df)
    
    # 2. Extract base values
    is_approved = int(preds[0])
    approved_prob = float(proba[0, 1])
    requested_amount = float(applicant_df['loan_amount'].iloc[0])
    
    # 3. Regression Stage (Run to determine eligible amount)
    applicant_df_reg = applicant_df.copy()
    applicant_df_reg['loan_status'] = 'Approve'
    reg_pred = float(reg.predict(applicant_df_reg)[0])
    
    # 4. Strict Validation Logic
    # Apply a small tolerance ($1.00) to handle floating-point precision differences
    tolerance = 1.00
    
    if is_approved == 1 and (requested_amount > (reg_pred + tolerance)):
        # Force reject if the user wants more than the model says they can afford
        is_approved = 0
        
    # 5. Final result assembly
    return [{
        'approved': is_approved,
        'approved_proba': approved_prob,
        'reg_pred': reg_pred
    }]