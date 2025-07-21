import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load the thyroid dataset
thyroid_df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing values
for column in thyroid_df.columns:
    if thyroid_df[column].dtype == 'object':
        thyroid_df[column].fillna(thyroid_df[column].mode()[0], inplace=True)
    else:
        thyroid_df[column].fillna(thyroid_df[column].mean(), inplace=True)

# Encode categorical variables
encoded_df = thyroid_df.copy()
for column in encoded_df.columns:
    if encoded_df[column].dtype == 'object':
        label_encoder = LabelEncoder()
        encoded_df[column] = label_encoder.fit_transform(encoded_df[column].astype(str))

# Select the first 20 rows for analysis
sample_df = encoded_df.iloc[:20].reset_index(drop=True)

# ---------- Cosine Similarity Heatmap ----------
cosine_similarity_matrix = cosine_similarity(sample_df)

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_similarity_matrix, annot=False, cmap="coolwarm")
plt.title("🔵 Cosine Similarity Heatmap (First 20 Observations)")
plt.tight_layout()
plt.show()

# ---------- JC and SMC Functions ----------
def calculate_jaccard_and_smc(vector1, vector2):
    f11 = np.sum((vector1 == 1) & (vector2 == 1))
    f00 = np.sum((vector1 == 0) & (vector2 == 0))
    f10 = np.sum((vector1 == 1) & (vector2 == 0))
    f01 = np.sum((vector1 == 0) & (vector2 == 1))

    jaccard_score = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
    smc_score = (f11 + f00) / (f00 + f01 + f10 + f11)
    return jaccard_score, smc_score

# Filter binary columns
binary_columns = [col for col in sample_df.columns if sample_df[col].isin([0, 1]).all()]

# Initialize JC and SMC matrices
jaccard_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))

# Compute pairwise JC and SMC values
for i in range(20):
    for j in range(20):
        vector_i = sample_df.loc[i, binary_columns].astype(int).values
        vector_j = sample_df.loc[j, binary_columns].astype(int).values
        jc_score, smc_score = calculate_jaccard_and_smc(vector_i, vector_j)
        jaccard_matrix[i, j] = jc_score
        smc_matrix[i, j] = smc_score

# ---------- Jaccard Coefficient Heatmap ----------
plt.figure(figsize=(10, 8))
sns.heatmap(jaccard_matrix, annot=False, cmap="viridis")
plt.title("🟢 Jaccard Coefficient Heatmap (Binary Features)")
plt.tight_layout()
plt.show()

# ---------- Simple Matching Coefficient Heatmap ----------
plt.figure(figsize=(10, 8))
sns.heatmap(smc_matrix, annot=False, cmap="YlGnBu")
plt.title("🟡 Simple Matching Coefficient Heatmap (Binary Features)")
plt.tight_layout()
plt.show()
