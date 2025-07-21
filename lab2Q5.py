import pandas as pd

# Load the thyroid dataset
thyroid_df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing categorical values with mode for now (only for binary comparison)
thyroid_df.fillna(thyroid_df.mode().iloc[0], inplace=True)

# Identify binary attributes (attributes having only 0 and 1 values)
binary_columns = [col for col in thyroid_df.columns if thyroid_df[col].dropna().value_counts().index.isin([0, 1]).all()]

print("Binary attributes used:", binary_columns)

# Select the first two observation rows using binary attributes
binary_vector_1 = thyroid_df.loc[0, binary_columns].astype(int)
binary_vector_2 = thyroid_df.loc[1, binary_columns].astype(int)

# Compute binary similarity components
f11 = ((binary_vector_1 == 1) & (binary_vector_2 == 1)).sum()
f00 = ((binary_vector_1 == 0) & (binary_vector_2 == 0)).sum()
f10 = ((binary_vector_1 == 1) & (binary_vector_2 == 0)).sum()
f01 = ((binary_vector_1 == 0) & (binary_vector_2 == 1)).sum()

# Calculate Jaccard Coefficient and Simple Matching Coefficient
jaccard_coeff = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
smc_coeff = (f11 + f00) / (f00 + f01 + f10 + f11)

# Display results
print(f"f11 = {f11}, f00 = {f00}, f10 = {f10}, f01 = {f01}")
print(f"Jaccard Coefficient = {jaccard_coeff}")
print(f"Simple Matching Coefficient = {smc_coeff}")

# Interpretation
if abs(jaccard_coeff - smc_coeff) > 0.1:
    print("Jaccard Coefficient focuses on shared presence (1s), better when 1s are more meaningful.")
else:
    print("Both JC and SMC are similar, but JC is preferred for sparse binary data.")
