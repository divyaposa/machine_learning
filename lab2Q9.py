import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the thyroid dataset
thyroid_df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing numeric values with column means temporarily for scaling
thyroid_df.fillna(thyroid_df.mean(numeric_only=True), inplace=True)

# Identify numeric columns
numeric_columns = thyroid_df.select_dtypes(include=[np.number]).columns.tolist()

# Display range of each numeric column
print("📊 Column Ranges:")
for column in numeric_columns:
    min_val = thyroid_df[column].min()
    max_val = thyroid_df[column].max()
    value_range = max_val - min_val
    print(f"{column}: Min = {min_val}, Max = {max_val}, Range = {value_range}")

# Initialize scalers
minmax_scaler = MinMaxScaler()
zscore_scaler = StandardScaler()

# Apply Min-Max Scaling
minmax_scaled_df = thyroid_df.copy()
minmax_scaled_df[numeric_columns] = minmax_scaler.fit_transform(thyroid_df[numeric_columns])
print("\n✅ Min-Max Scaling Applied")

# Apply Z-score Standardization
zscore_scaled_df = thyroid_df.copy()
zscore_scaled_df[numeric_columns] = zscore_scaler.fit_transform(thyroid_df[numeric_columns])
print("✅ Z-score Standardization Applied")

# Preview few rows of scaled data
print("\n📈 Sample - Min-Max Scaled Data:\n", minmax_scaled_df[numeric_columns].head())
print("\n📉 Sample - Z-score Standardized Data:\n", zscore_scaled_df[numeric_columns].head())
