import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the thyroid dataset
thyroid_df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing values based on type
for column in thyroid_df.columns:
    if thyroid_df[column].dtype == 'object':
        thyroid_df[column].fillna(thyroid_df[column].mode()[0], inplace=True)
    else:
        thyroid_df[column].fillna(thyroid_df[column].mean(), inplace=True)

# Encode all categorical variables using Label Encoding
encoded_df = thyroid_df.copy()
for column in encoded_df.columns:
    if encoded_df[column].dtype == 'object':
        label_encoder = LabelEncoder()
        encoded_df[column] = label_encoder.fit_transform(encoded_df[column].astype(str))

# Extract the first two observations
observation_vector_1 = encoded_df.iloc[0].values.reshape(1, -1)
observation_vector_2 = encoded_df.iloc[1].values.reshape(1, -1)

# Compute cosine similarity
cosine_score = cosine_similarity(observation_vector_1, observation_vector_2)[0][0]

print(f"Cosine Similarity between the first two observations: {cosine_score:.4f}")
