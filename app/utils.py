import pandas as pd

def build_applicant_from_dict(d,expected_columns):
    df = pd.DataFrame([d])
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].str.strip()
    missing = [c for c in expected_columns if c not in df.columns]
    # print(expected_columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[expected_columns]
# build_applicant_from_dict(data)