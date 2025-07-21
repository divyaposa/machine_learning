import pandas as pd
import numpy as np

# Load the thyroid dataset
thyroid_df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Identify numeric and categorical columns
numeric_columns = thyroid_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = thyroid_df.select_dtypes(include=['object']).columns.tolist()

# -----------------------------
# Imputation for numeric columns
# -----------------------------
for column in numeric_columns:
    if thyroid_df[column].isnull().sum() == 0:
        continue

    # Calculate IQR to detect outliers
    q1 = thyroid_df[column].quantile(0.25)
    q3 = thyroid_df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    has_outliers = ((thyroid_df[column] < lower_bound) | (thyroid_df[column] > upper_bound)).any()
    
    # Choose imputation method
    if has_outliers:
        imputed_value = thyroid_df[column].median()
        strategy = "median"
    else:
        imputed_value = thyroid_df[column].mean()
        strategy = "mean"
    
    thyroid_df[column].fillna(imputed_value, inplace=True)
    print(f"{column}: Filled missing with {strategy} = {imputed_value}")

# -----------------------------
# Imputation for categorical columns using mode
# -----------------------------
for column in categorical_columns:
    if thyroid_df[column].isnull().sum() == 0:
        continue

    mode_value = thyroid_df[column].mode()[0]
    thyroid_df[column].fillna(mode_value, inplace=True)
    print(f"{column}: Filled missing with mode = {mode_value}")

# -----------------------------
# Confirm missing values filled
# -----------------------------
print("\n✅ Missing values after imputation:\n", thyroid_df.isnull().sum())
