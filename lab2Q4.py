import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the thyroid dataset
# -----------------------------
thyroid_df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# View the first few rows
print("🔍 Sample Data:\n", thyroid_df.head())

# -----------------------------
# 2. Data types of each attribute
# -----------------------------
print("\n📊 Data Types:\n", thyroid_df.dtypes)

# -----------------------------
# 3. Check for missing values
# -----------------------------
print("\n❓ Missing Values:\n", thyroid_df.isnull().sum())

# -----------------------------
# 4. Detect categorical columns
# -----------------------------
categorical_columns = thyroid_df.select_dtypes(include=['object']).columns.tolist()
print("\n🗂️ Categorical Attributes:", categorical_columns)

# 💡 Encoding Scheme Hint
print("\n💡 Encoding Recommendation:")
print("- Use Label Encoding for ordinal attributes (e.g., severity levels)")
print("- Use One-Hot Encoding for nominal attributes (e.g., gender, class)")

# -----------------------------
# 5. Apply Label Encoding as placeholder
# -----------------------------
label_encoded_df = thyroid_df.copy()
for column in categorical_columns:
    label_encoder = LabelEncoder()
    try:
        label_encoded_df[column] = label_encoder.fit_transform(thyroid_df[column].astype(str))
    except:
        print(f"Skipping encoding for {column}")

# -----------------------------
# 6. Analyze numeric columns
# -----------------------------
numeric_columns = thyroid_df.select_dtypes(include=[np.number]).columns.tolist()
print("\n📈 Numeric Summary:")
for column in numeric_columns:
    mean_val = thyroid_df[column].mean()
    std_val = thyroid_df[column].std()
    var_val = thyroid_df[column].var()
    min_val = thyroid_df[column].min()
    max_val = thyroid_df[column].max()
    print(f"{column}: Mean = {mean_val:.2f}, Std = {std_val:.2f}, Var = {var_val:.2f}, Range = {min_val} to {max_val}")

# -----------------------------
# 7. Outlier detection using boxplot (Filtered)
# -----------------------------
# Remove identifier-like columns that may distort plots
excluded_columns = ['Record ID', 'record_id', 'id']
plot_numeric_columns = [col for col in numeric_columns if col not in excluded_columns]

# Plot boxplot for 'age' with upper bound filter for better visualization
plt.figure(figsize=(6, 4))
thyroid_df[thyroid_df['age'] < 200]['age'].plot.box()
plt.title("📦 Boxplot of Age (Filtered for age < 200)")
plt.grid(True)
plt.tight_layout()
plt.show()
